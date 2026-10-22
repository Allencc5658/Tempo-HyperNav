import numpy as np
import quaternion


class RealWorld_Env:
    def __init__(self, args):
        self.args = args
        self.episode_over = False
        self.current_episode = None
    
    def seed(self, seed):
        return

    def reset(self):
        return self.get_observation()

    def step(self, action):
        # TODO: implement the step function
        return self.get_observation()

    def get_observation(self):
        # TODO: get the rgbd from the real world, and the compass and gps of the camera
        observation = {}
        observation['rgb'] = np.zeros((3, 224, 224), dtype=np.uint8)
        observation['depth'] = np.zeros((1, 224, 224), dtype=np.float32)
        observation['compass'] = np.array([0.], dtype=np.float32)
        observation['gps'] = np.array([0., 0.], dtype=np.float32)
        return observation

    def get_metrics(self):
        metrics = {}
        metrics['spl'] = 0.
        metrics['success'] = 0.
        metrics['distance_to_goal'] = 0.
        metrics['soft_spl'] = 0.
        return metrics

    def get_agent_pose(self):
        # TODO: get the position and rotation of the robot from the real world
        position = np.array([0., 0., 0.], dtype=np.float32)
        rotation = quaternion.quaternion(1., 0., 0., 0.)
        return {'position': position, 'rotation': rotation}
