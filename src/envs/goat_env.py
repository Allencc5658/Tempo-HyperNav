"""GOAT-Bench long-horizon adapter for UniGoal.

The released GOAT files are replayed directly while Habitat supplies only the
HM3D simulator.  Scene memory is retained across goals; only goal-specific
state is reset by the caller when ``subtask_idx`` changes.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import quaternion
import torch
from omegaconf import OmegaConf

from configs.categories import name2index
from src.utils.fmm.pose_utils import get_rel_pose_change
from .goat_episode_parser import (
    GOAL_TYPE_DESCRIPTION,
    GOAL_TYPE_IMAGE,
    GOAL_TYPE_OBJECT,
    GoatEpisodeParser,
)


class GoatEnv:
    """Expose GOAT episodes through the interface used by ``UniGoal_Agent``."""

    def __init__(self, args, config_env, dataset):
        import habitat

        self._env = habitat.Env(config_env, dataset)
        self.args = args
        self.device = torch.device("cuda" if getattr(args, "cuda", True) else "cpu")
        self._base_sim_config = copy.deepcopy(config_env.habitat.simulator)

        dataset_path = args.goat_dataset_path.replace("{split}", args.goat_split)
        self.goat_parser = GoatEpisodeParser(dataset_path)
        self.goat_categories = list(self.goat_parser.categories)

        self.scene_data_path = os.path.abspath(
            getattr(args, "goat_scene_data_path", "data/scene_datasets/hm3d_v0.2/val")
        )
        self._validate_scene_files()

        self.name2index = name2index
        self.index2name = {value: key for key, value in self.name2index.items()}
        self.episode_no = getattr(args, "start_episode", 0) - 1
        self.goat_episode_idx = getattr(args, "start_episode", 0) - 1
        self.current_goat_episode = None
        self.subtask_idx = 0

        self.timestep = 0
        self.episode_timestep = 0
        self.path_length = 0.0
        self._subtask_path_length = 0.0
        self._subtask_start_pos = None
        self._subtask_shortest_distance = float("inf")
        self._last_agent_position = None
        self.last_agent_location = None
        self._last_obs = None
        self._subtask_results: List[Dict] = []

        self.goal_name = None
        self.gt_goal_idx = -1
        self.goal_object_id = -1
        self._goal_object_ids = []
        self.instance_imagegoal = None
        self.text_goal = None
        self.goal_type = None

        self.success_distance = float(getattr(args, "success_dist", 1.0))
        self.subtask_budget_mode = getattr(args, "subtask_budget_mode", "goat_discrete_actions")
        if self.subtask_budget_mode != "goat_discrete_actions":
            raise ValueError(
                "This adapter executes Habitat discrete actions; supported GOAT budget "
                "mode is 'goat_discrete_actions'. MSGNav native planner steps require "
                "its pose-based controller and cannot be counted as discrete actions."
            )
        self.max_subtask_steps = int(
            getattr(args, "max_subtask_actions", getattr(args, "max_episode_length", 500))
        )
        self.max_episode_steps = int(getattr(args, "max_episode_steps", 5000))
        self.start_height = 0.0
        self.agent_height = float(args.camera_height)
        self.info = {"distance_to_goal": None, "spl": None, "success": None}

    # ------------------------------------------------------------------
    # Public environment interface
    # ------------------------------------------------------------------

    def reset(self):
        self.episode_no += 1
        self.goat_episode_idx += 1
        if self.goat_episode_idx >= len(self.goat_parser):
            raise StopIteration("All GOAT episodes have been evaluated.")

        self.current_goat_episode = self.goat_parser[self.goat_episode_idx]
        self.subtask_idx = 0
        self.timestep = 0
        self.episode_timestep = 0
        self.path_length = 0.0
        self._subtask_results = []

        # Reset Habitat's task bookkeeping, then replace its unrelated dataset
        # scene with the scene referenced by the GOAT episode.
        self._env.reset()
        self._reconfigure_scene(self.current_goat_episode.scene_id)
        self._env.sim.set_agent_state(
            self.current_goat_episode.start_position,
            self.current_goat_episode.start_rotation,
        )
        obs = self._env.sim.get_observations_at()
        obs = self._augment_episode_sensors(obs)
        self._last_obs = obs

        pose = self.get_agent_pose()
        self.start_height = float(pose["position"][1])
        self.agent_height = float(self.args.camera_height)
        self.last_agent_location = self.get_agent_location()
        self._last_agent_position = np.asarray(pose["position"], dtype=np.float64)
        self._init_subtask()

        print(
            f"[GOAT] episode={self.episode_no} scene={Path(self._resolved_scene_path()).name} "
            f"subtask=0/{len(self.current_goat_episode.subtasks)} "
            f"type={self.goal_type} category={self.goal_name}"
        )
        torch.set_grad_enabled(True)
        return obs, self.info

    def step(self, action):
        action_int = int(action["action"] if isinstance(action, dict) else action)

        # Habitat's StopAction terminates its underlying (unrelated) episode.
        # GOAT STOP is therefore handled by this adapter without forwarding it.
        if action_int == 0:
            obs = self._env.sim.get_observations_at()
        else:
            if self._env.episode_over:
                raise RuntimeError(
                    "Habitat ended before the GOAT chain. Check max_episode_steps >= 5000."
                )
            obs = self._env.step(action)
        obs = self._augment_episode_sensors(obs)
        self._last_obs = obs

        self.timestep += 1
        self.episode_timestep += 1
        self._track_motion()

        agent_pose = self.get_agent_pose()
        self.agent_height = (
            float(self.args.camera_height)
            + float(agent_pose["position"][1])
            - self.start_height
        )

        subtask_done = action_int == 0
        timed_out = self.timestep >= self.max_subtask_steps
        chain_timed_out = self.episode_timestep >= self.max_episode_steps
        if subtask_done or timed_out or chain_timed_out:
            self._finish_subtask()
            next_subtask = self.subtask_idx + 1
            episode_done = chain_timed_out or next_subtask >= len(
                self.current_goat_episode.subtasks
            )
            if episode_done:
                self._finalize_episode_metrics()
            else:
                self.subtask_idx = next_subtask
                self._init_subtask()
                print(
                    f"[GOAT] subtask={self.subtask_idx}/"
                    f"{len(self.current_goat_episode.subtasks)} "
                    f"type={self.goal_type} category={self.goal_name}"
                )
            return obs, episode_done, self.info

        self.info["time"] = self.timestep
        self.info["sensor_pose"] = self._sensor_pose
        self.info["agent_height"] = self.agent_height
        return obs, False, self.info

    def seed(self, seed):
        self._env.seed(seed)

    def set_goal_cat_id(self, idx):
        if idx is None:
            return
        self.gt_goal_idx = int(idx)
        self.info["goal_cat_id"] = self.gt_goal_idx

    def get_metrics(self):
        spl = float(self.info.get("spl") or 0.0)
        success = float(self.info.get("success") or 0.0)
        distance = float(self.info.get("distance_to_goal") or 0.0)
        return spl, success, distance, float(self.info.get("soft_spl", spl))

    def get_goat_metrics(self):
        results = [dict(result) for result in self._subtask_results]
        successes = [float(result["success"]) for result in results]
        spls = [float(result["spl"]) for result in results]
        by_type = {GOAL_TYPE_OBJECT: [], GOAL_TYPE_DESCRIPTION: [], GOAL_TYPE_IMAGE: []}
        for result in results:
            by_type.setdefault(result["goal_type"], []).append(result)
        return {
            "subtask_results": results,
            "subtask_sr": successes,
            "subtask_spl": spls,
            "chain_sr": float(np.mean(successes)) if successes else 0.0,
            "episode_sr": float(all(successes)) if successes else 0.0,
            "mean_spl": float(np.mean(spls)) if spls else 0.0,
            "sr_by_type": {
                key: float(np.mean([item["success"] for item in values])) if values else 0.0
                for key, values in by_type.items()
            },
            "spl_by_type": {
                key: float(np.mean([item["spl"] for item in values])) if values else 0.0
                for key, values in by_type.items()
            },
        }

    # ------------------------------------------------------------------
    # Scene and observation handling
    # ------------------------------------------------------------------

    def _scene_relative_path(self, scene_id=None):
        scene_id = scene_id or self.current_goat_episode.scene_id
        parts = [part for part in scene_id.replace("\\", "/").split("/") if part]
        if "val" in parts:
            parts = parts[parts.index("val") + 1 :]
        if len(parts) < 2:
            raise ValueError(f"Unrecognised GOAT scene id: {scene_id}")
        return os.path.join(parts[-2], parts[-1])

    def _resolved_scene_path(self, scene_id=None):
        return os.path.abspath(os.path.join(self.scene_data_path, self._scene_relative_path(scene_id)))

    def _validate_scene_files(self):
        missing = []
        for episode in self.goat_parser.episodes:
            scene = self._resolved_scene_path(episode.scene_id)
            navmesh = os.path.splitext(scene)[0] + ".navmesh"
            if not os.path.isfile(scene) or not os.path.isfile(navmesh):
                missing.append((scene, navmesh))
        if missing:
            first = missing[0]
            raise FileNotFoundError(
                f"Missing HM3D v0.2 scene/navmesh for {len(missing)} GOAT scenes; first={first}"
            )
        print(
            f"[GOAT] Validated {len({ep.scene_id for ep in self.goat_parser.episodes})} "
            f"HM3D v0.2 scenes under {self.scene_data_path}"
        )

    def _reconfigure_scene(self, scene_id):
        sim_config = copy.deepcopy(self._base_sim_config)
        OmegaConf.set_readonly(sim_config, False)
        sim_config.scene = self._resolved_scene_path(scene_id)
        self._env.sim.reconfigure(sim_config)

    @staticmethod
    def _xyzw_quaternion(coefficients):
        x, y, z, w = [float(value) for value in coefficients]
        return quaternion.quaternion(w, x, y, z)

    def _augment_episode_sensors(self, observations):
        observations = dict(observations or {})
        state = self._env.sim.get_agent_state(0)
        episode = self.current_goat_episode
        start_rotation = self._xyzw_quaternion(episode.start_rotation)
        relative_position = quaternion.rotate_vectors(
            start_rotation.inverse(),
            np.asarray(state.position) - np.asarray(episode.start_position),
        )
        observations["gps"] = np.asarray(
            [-relative_position[2], relative_position[0]], dtype=np.float32
        )
        relative_rotation = state.rotation.inverse() * start_rotation
        heading_vector = quaternion.rotate_vectors(
            relative_rotation, np.asarray([0.0, 0.0, -1.0])
        )
        observations["compass"] = np.asarray(
            [np.arctan2(heading_vector[0], -heading_vector[2])], dtype=np.float32
        )
        return observations

    def _render_image_goal(self, image_goal):
        import habitat_sim
        import habitat_sim._ext.habitat_sim_bindings as hsim
        from habitat.utils.geometry_utils import quaternion_from_coeff
        from habitat_sim.agent.agent import AgentState, SixDOFPose

        sensor_uuid = f"goat_image_goal_{self.episode_no}_{self.subtask_idx}"
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = sensor_uuid
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = image_goal["image_dimensions"]
        spec.hfov = float(image_goal["hfov"])
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        self._env.sim.add_sensor(spec)
        agent = self._env.sim.get_agent(0)
        previous_state = agent.get_state()
        try:
            agent.set_state(
                AgentState(
                    position=previous_state.position,
                    rotation=previous_state.rotation,
                    sensor_states={
                        **previous_state.sensor_states,
                        sensor_uuid: SixDOFPose(
                            position=np.asarray(image_goal["position"]),
                            rotation=quaternion_from_coeff(image_goal["rotation"]),
                        ),
                    },
                ),
                infer_sensor_states=False,
            )
            self._env.sim._sensors[sensor_uuid].draw_observation()
            return self._env.sim._sensors[sensor_uuid].get_observation()[:, :, :3]
        finally:
            del self._env.sim._sensors[sensor_uuid]
            hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
            del agent._sensors[sensor_uuid]
            agent.agent_config.sensor_specifications = [
                item for item in agent.agent_config.sensor_specifications
                if item.uuid != sensor_uuid
            ]

    # ------------------------------------------------------------------
    # Subtask goals and MSGNav-compatible metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _viewpoint_positions(view_points: Iterable[dict]):
        return [
            np.asarray(viewpoint.get("agent_state", {}).get("position"), dtype=np.float32)
            for viewpoint in view_points
        ]

    def _geodesic_distance(self, start, view_points):
        positions = self._viewpoint_positions(view_points)
        if not positions:
            return float("inf")
        distance = float(self._env.sim.geodesic_distance(np.asarray(start), positions))
        return distance if np.isfinite(distance) else float("inf")

    def _init_subtask(self):
        task = self.current_goat_episode.subtasks[self.subtask_idx]
        self.goal_type = task.goal_type
        self.goal_name = task.object_category
        self.goal_object_id = task.object_id if task.object_id is not None else -1
        self._goal_object_ids = [
            str(goal.get("object_id")) for goal in task.goal_instances
            if goal.get("object_id") is not None
        ]
        self.gt_goal_idx = self.name2index.get(self.goal_name, -1)
        self.instance_imagegoal = None
        self.text_goal = None
        if task.goal_type == GOAL_TYPE_IMAGE:
            self.instance_imagegoal = self._render_image_goal(task.image_goal)
        elif task.goal_type == GOAL_TYPE_DESCRIPTION:
            self.text_goal = task.lang_desc

        current_position = np.asarray(self.get_agent_pose()["position"], dtype=np.float64)
        self._subtask_start_pos = current_position.copy()
        self._last_agent_position = current_position.copy()
        self._subtask_path_length = 0.0
        self._subtask_shortest_distance = self._geodesic_distance(
            current_position, task.view_points
        )
        if np.isfinite(self._subtask_shortest_distance):
            self._subtask_shortest_distance += 1e-6
        self.timestep = 0
        self._update_info_for_subtask()

    def _track_motion(self):
        current_position = np.asarray(self.get_agent_pose()["position"], dtype=np.float64)
        # MSGNav logger accumulates movement in its 2-D TSDF voxel plane.
        displacement = float(
            np.linalg.norm(current_position[[0, 2]] - self._last_agent_position[[0, 2]])
        )
        self.path_length += displacement
        self._subtask_path_length += displacement
        self._last_agent_position = current_position
        dx, dy, do = get_rel_pose_change(self.get_agent_location(), self.last_agent_location)
        self.last_agent_location = self.get_agent_location()
        self._sensor_pose = [dx, dy, do]

    def _finish_subtask(self):
        task = self.current_goat_episode.subtasks[self.subtask_idx]
        final_distance = self._geodesic_distance(
            self.get_agent_pose()["position"], task.view_points
        )
        # MSGNav evaluates the final geodesic distance after its loop; it does
        # not require a Habitat STOP action for the distance-based SR.
        success = bool(final_distance < self.success_distance)
        shortest = self._subtask_shortest_distance
        actual = self._subtask_path_length
        spl = 0.0
        if success and np.isfinite(shortest):
            spl = float(shortest / max(shortest, actual, 1e-8))
        self._subtask_results.append(
            {
                "subtask_idx": self.subtask_idx,
                "goal_type": self.goal_type,
                "category": self.goal_name,
                "object_id": None if self.goal_object_id == -1 else self.goal_object_id,
                "success": float(success),
                "spl": spl,
                "distance_to_goal": final_distance,
                "shortest_path": shortest,
                "path_length": actual,
                "num_steps": self.timestep,
            }
        )

    def _update_info_for_subtask(self):
        self.info.update(
            {
                "goal_cat_id": self.gt_goal_idx,
                "goal_name": self.goal_name,
                "goal_type": self.goal_type,
                "subtask_idx": self.subtask_idx,
                "num_subtasks": len(self.current_goat_episode.subtasks),
                "episode_no": self.episode_no,
                "sensor_pose": [0.0, 0.0, 0.0],
                "agent_height": self.agent_height,
                "time": 0,
            }
        )
        self.info.pop("instance_imagegoal", None)
        self.info.pop("text_goal", None)
        if self.goal_type == GOAL_TYPE_IMAGE:
            self.info["instance_imagegoal"] = self.instance_imagegoal
        elif self.goal_type == GOAL_TYPE_DESCRIPTION:
            self.info["text_goal"] = self.text_goal

    def _finalize_episode_metrics(self):
        metrics = self.get_goat_metrics()
        last_distance = (
            self._subtask_results[-1]["distance_to_goal"] if self._subtask_results else 0.0
        )
        self.info.update(
            {
                "success": metrics["chain_sr"],
                "episode_success": metrics["episode_sr"],
                "spl": metrics["mean_spl"],
                "soft_spl": metrics["mean_spl"],
                "distance_to_goal": last_distance,
                "goat_metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # Pose helpers kept compatible with InstanceImageGoal_Env
    # ------------------------------------------------------------------

    def get_agent_pose(self):
        state = self._env.sim.get_agent_state(0)
        return {"position": state.position, "rotation": state.rotation}

    def get_agent_location(self):
        pose = self.get_agent_pose()
        x = -pose["position"][2]
        y = -pose["position"][0]
        euler = quaternion.as_euler_angles(pose["rotation"])
        axis = euler[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            orientation = euler[1]
        else:
            orientation = 2 * np.pi - euler[1]
        if orientation > np.pi:
            orientation -= 2 * np.pi
        return x, y, orientation
