import os
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from .instanceimagegoal_env import InstanceImageGoal_Env


def _build_habitat_config(args):
    """Shared helper: build and return a habitat config + dataset for the given args."""
    from habitat.config.default import get_config
    from habitat import make_dataset

    basic_config = get_config(config_path="configs/" + args.task_config)
    OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split

    OmegaConf.set_readonly(basic_config, True)

    dataset = make_dataset(basic_config.habitat.dataset.type)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)

    scene_split_sizes = []
    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    env_idx = 0
    config_env = get_config(config_path="configs/" + args.task_config)
    OmegaConf.set_readonly(config_env, False)

    if len(scenes) > 0:
        contentss = scenes[
            sum(scene_split_sizes[:env_idx]):
            sum(scene_split_sizes[:env_idx + 1])
        ]
        bad_scense = []
        good_scense = [contentss[i] for i in range(len(contentss))
                       if contentss[i] not in bad_scense]
        config_env.habitat.dataset.content_scenes = good_scense

    gpu_id = getattr(args, "sim_gpu_id", 0)
    config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
    config_env.habitat.environment.iterator_options.shuffle = False

    agent_config = config_env.habitat.simulator.agents.main_agent
    sim_sensors = agent_config.sim_sensors

    sim_sensors.rgb_sensor.width = args.env_frame_width
    sim_sensors.rgb_sensor.height = args.env_frame_height
    sim_sensors.rgb_sensor.hfov = args.hfov
    sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]
    if hasattr(args, "camera_tilt_deg"):
        camera_orientation = [
            float(np.deg2rad(args.camera_tilt_deg)),
            0.0,
            0.0,
        ]
        sim_sensors.rgb_sensor.orientation = camera_orientation

    sim_sensors.depth_sensor.width = args.env_frame_width
    sim_sensors.depth_sensor.height = args.env_frame_height
    sim_sensors.depth_sensor.hfov = args.hfov
    sim_sensors.depth_sensor.min_depth = args.min_depth
    sim_sensors.depth_sensor.max_depth = args.max_depth
    sim_sensors.depth_sensor.position = [0, args.camera_height, 0]
    if hasattr(args, "camera_tilt_deg"):
        sim_sensors.depth_sensor.orientation = camera_orientation

    if "semantic_sensor" in sim_sensors:
        sim_sensors.semantic_sensor.width = args.env_frame_width
        sim_sensors.semantic_sensor.height = args.env_frame_height
        sim_sensors.semantic_sensor.hfov = args.hfov
        sim_sensors.semantic_sensor.position = [0, args.camera_height, 0]
        if hasattr(args, "camera_tilt_deg"):
            sim_sensors.semantic_sensor.orientation = camera_orientation

    agent_config.height = getattr(args, "agent_height", args.camera_height)
    if hasattr(args, "agent_radius"):
        agent_config.radius = args.agent_radius
    config_env.habitat.simulator.turn_angle = args.turn_angle
    if hasattr(args, "forward_step_size"):
        config_env.habitat.simulator.forward_step_size = args.forward_step_size
    if hasattr(args, "allow_sliding"):
        config_env.habitat.simulator.habitat_sim_v0.allow_sliding = args.allow_sliding
    if hasattr(args, "max_episode_steps"):
        config_env.habitat.environment.max_episode_steps = args.max_episode_steps
    if "success" in config_env.habitat.task.measurements:
        config_env.habitat.task.measurements.success.success_distance = args.success_dist
    config_env.habitat.dataset.split = args.split

    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    start_episode = getattr(args, 'start_episode', 0)
    if start_episode > 0 and getattr(args, 'goal_type', '') != 'goat':
        if start_episode >= len(dataset.episodes):
            raise ValueError(
                f"start_episode ({start_episode}) >= total episodes ({len(dataset.episodes)}). "
                "Evaluation is already complete.")
        dataset.episodes = dataset.episodes[start_episode:]

    OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    return config_env, dataset


def construct_envs(args):
    if args.environment == 'habitat':
        if getattr(args, 'goal_type', '') == 'goat':
            from .goat_env import GoatEnv
            config_env, dataset = _build_habitat_config(args)
            env = GoatEnv(args=args, config_env=config_env, dataset=dataset)
            env.seed(0)
            return env
        else:
            config_env, dataset = _build_habitat_config(args)
            env = InstanceImageGoal_Env(args=args, config_env=config_env, dataset=dataset)
            env.seed(0)
            return env
    elif args.environment == 'real_world':
        env = InstanceImageGoal_Env(args=args)
        return env
