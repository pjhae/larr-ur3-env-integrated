import numpy as np
import os
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv


class DualUR3Env(MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_filename = 'dual_ur3_larr.xml'
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'ur3', xml_filename)
        MujocoEnv.__init__(self, fullpath, 1)
        self.init_qpos[0:6] = np.array([-90.0, -90.0, -90.0, -45.0, 225.0, 0.0])*np.pi/180.0 # right arm
        self.init_qpos[6:12] = np.array([90.0, -90.0, 90.0, 225.0, 135.0, 0.0])*np.pi/180.0 # left arm

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent