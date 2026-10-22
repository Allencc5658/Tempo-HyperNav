import json
import gzip
import gym
import numpy as np
import quaternion
import os
import torch
import numpy as np
import json
from PIL import Image

from configs.categories import name2index
from src.utils.fmm.pose_utils import get_l2_distance, get_rel_pose_change


class InstanceImageGoal_Env:
    def __init__(self, args, config_env=None, dataset=None):
        if args.environment == 'habitat':
            import habitat
            self._env = habitat.Env(config_env, dataset)
        elif args.environment == 'real_world':
            from .real_world_env import RealWorld_Env
            self._env = RealWorld_Env(args)
        self.args = args

        self.device = torch.device("cuda")
        self.episode_no = getattr(args, 'start_episode', 0) - 1

        # Episode Dataset info
        self.goal_name = None

        # Episode tracking info
        self.timestep = None
        self.stopped = None
        self.last_action = None
        self.path_length = None
        self.last_agent_location = None
        self.trajectory_states = []
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None

        self.name2index = name2index
        self.index2name = {v: k for k, v in self.name2index.items()}
        if self.args.goal_type == 'text':
            with gzip.open(self.args.text_goal_dataset, 'rt') as f:
                self.text_goal_dataset = json.load(f)
            self.average_acc = 0

    def update_after_reset(self):
        goal_name = self._env.current_episode.object_category

        self.goal_name = goal_name
        self.gt_goal_idx = self.name2index[goal_name]

        # InstanceImageGoal / TextGoal: single specific instance
        # ObjectNav: all instances of the goal category
        if hasattr(self._env.current_episode, 'goal_object_id'):
            self.goal_object_id = int(self._env.current_episode.goal_object_id)
            self._goal_object_ids = [self.goal_object_id]
        elif hasattr(self._env.current_episode, 'goals') and self._env.current_episode.goals:
            self._goal_object_ids = []
            for g in self._env.current_episode.goals:
                try:
                    self._goal_object_ids.append(int(g.object_id))
                except (ValueError, TypeError):
                    pass
            self.goal_object_id = self._goal_object_ids[0] if self._goal_object_ids else -1
        else:
            self.goal_object_id = -1
            self._goal_object_ids = []

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.last_action = None
        self.path_length = 1e-5
        self.trajectory_states = []

        if self.args.episode_id != -1:
            if self.args.environment == 'habitat':
                self._env.current_episode = self._env.episodes[self.args.episode_id]
            self.episode_no = self.args.episode_id
       
        obs = self._env.reset()
        self.update_after_reset()
        if 'semantic' in obs:
            semantic_obs = obs['semantic']
            sem = np.isin(semantic_obs, self._goal_object_ids).astype(int)
            self.semantic_obs = sem
            self.sign = np.any(sem > 0)

        self.last_agent_location = self.get_agent_location()
        # upstair or downstair check
        agent_pose = self.get_agent_pose()
        self.start_height = agent_pose['position'][1]
        self.agent_height = self.args.camera_height

        self.start_position = agent_pose['position']
        self.start_rotation = agent_pose['rotation']
            
        torch.set_grad_enabled(False)

        self.info['goal_cat_id'] = self.gt_goal_idx
        if self.args.goal_type == 'ins-image':
            if self.args.goal:
                instance_imagegoal_file = self.args.goal
                self.info['instance_imagegoal'] = np.array(Image.open(instance_imagegoal_file))
            else:
                self.info['instance_imagegoal'] = obs['instance_imagegoal']
            self.instance_imagegoal = self.info['instance_imagegoal']
        if self.args.goal_type == 'text':
            if self.args.goal:
                text_goal = self.args.goal
                self.info['text_goal'] = text_goal
            else:
                self.info['text_goal'] = self.text_goal_dataset['attribute_data'][self._env.current_episode.goal_key]
            self.text_goal = self.info['text_goal']

        print(f"episode:{self.episode_no}, cat_id:{self.gt_goal_idx}, cat_name:{self.goal_name}")
        torch.set_grad_enabled(True)

        # Set info
        self.info['time'] = self.timestep
        self.info['num_steps'] = self.timestep
        self.info['path_length'] = float(self.path_length)
        self.info['stopped'] = bool(self.stopped)
        self.info['last_action'] = self.last_action
        if hasattr(self.args, 'max_episode_steps'):
            self.info['max_episode_steps'] = int(self.args.max_episode_steps)
        elif hasattr(self.args, 'max_episode_length'):
            self.info['max_episode_steps'] = int(self.args.max_episode_length)
        self.info['sensor_pose'] = [0., 0., 0.]
        self.info['goal_cat_id'] = self.gt_goal_idx
        self.info['goal_name'] = self.goal_name
        self.info['agent_height'] = self.agent_height
        self.info['episode_no'] = self.episode_no
        
        return obs, self.info
    
    def seed(self, seed):
        self._env.seed(seed)

    def set_goal_cat_id(self, idx):
        self.gt_goal_idx = idx
        self.info['goal_cat_id'] = idx
        self.info['goal_name'] = self.index2name[idx]

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action_id = self._extract_action_id(action)
        self.last_action = action_id
        if action_id == 0:
            self.stopped = True

        obs = self._env.step(action)
        done = self._env.episode_over
    
        if 'semantic' in obs:
            semantic_obs = obs['semantic']
            sem = np.isin(semantic_obs, self._goal_object_ids).astype(int)
            self.semantic_obs = sem
            self.sign = np.any(sem > 0)

        agent_pose = self.get_agent_pose()
        self.agent_height = self.args.camera_height + agent_pose['position'][1] - self.start_height
        self.info['agent_height'] = self.agent_height

        # Get location change
        dx, dy, do = self.get_location_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += get_l2_distance(0, dx, 0, dy)

        if done:
            if self.args.goal:
                spl, success, dist, soft_spl = 0., 0., 0., 0.
            else:
                spl, success, dist, soft_spl = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
            self.info['soft_spl'] = soft_spl

        self.timestep += 1
        self.info['time'] = self.timestep
        self.info['num_steps'] = self.timestep
        self.info['path_length'] = float(self.path_length)
        self.info['stopped'] = bool(self.stopped)
        self.info['last_action'] = self.last_action
        if hasattr(self.args, 'max_episode_steps'):
            self.info['max_episode_steps'] = int(self.args.max_episode_steps)
        elif hasattr(self.args, 'max_episode_length'):
            self.info['max_episode_steps'] = int(self.args.max_episode_length)

        return obs, done, self.info

    @staticmethod
    def _extract_action_id(action):
        if isinstance(action, dict):
            action = action.get('action')
        try:
            return int(action)
        except (TypeError, ValueError):
            return None

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        metrics = self._env.get_metrics()
        spl, success, dist = metrics['spl'], metrics['success'], metrics['distance_to_goal']
        soft_spl = metrics['soft_spl']
        return spl, success, dist, soft_spl

    def get_agent_pose(self):
        if self.args.environment == 'habitat':
            agent_state = self._env.sim.get_agent_state(0)
            return {'position': agent_state.position, 'rotation': agent_state.rotation}
        elif self.args.environment == 'real_world':
            return self._env.get_agent_pose()

    def get_agent_location(self):
        """Returns x, y, o pose of the agent."""

        agent_pose = self.get_agent_pose()
        x = -agent_pose['position'][2]
        y = -agent_pose['position'][0]
        axis = quaternion.as_euler_angles(agent_pose['rotation'])[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_pose['rotation'])[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_pose['rotation'])[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_location_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_agent_location = self.get_agent_location()
        dx, dy, do = get_rel_pose_change(
            curr_agent_location, self.last_agent_location)
        self.last_agent_location = curr_agent_location
        return dx, dy, do
