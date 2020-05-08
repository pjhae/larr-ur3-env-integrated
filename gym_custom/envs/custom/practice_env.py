import numpy as np
import os
from gym_custom import utils
from gym_custom.envs.mujoco import MujocoEnv


class PracticeEnv1(MujocoEnv, utils.EzPickle):
    # InvertedPendulum from gym/envs/mujoco

    def __init__(self, actuator_type):
        utils.EzPickle.__init__(self)
        actuator_type_list = ['motor', 'position', 'position(dyntype=integrator)', 'velocity']
        if actuator_type == 'motor':
            xml_filename = 'practice1_motor.xml'
        elif actuator_type == 'position':
            xml_filename = 'practice1_position.xml'
        elif actuator_type == 'position(dyntype=integrator)':
            xml_filename = 'practice1_position_dyntype_integrator.xml'
        elif actuator_type == 'velocity':
            xml_filename = 'practice1_velocity.xml'
        else:
            raise ValueError('actuator_type not in %s'%actuator_type_list)
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
        MujocoEnv.__init__(self, fullpath, 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
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


class PracticeEnv2(MujocoEnv, utils.EzPickle):
    # InvertedDoublePendulumEnv from gym/envs/mujoco

    def __init__(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'practice2.xml')
        MujocoEnv.__init__(self, fullpath, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]