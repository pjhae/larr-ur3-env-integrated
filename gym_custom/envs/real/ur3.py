import time

import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, convert_action_to_space, convert_observation_to_space, COMMAND_LIMITS
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

class UR3RealEnv(gym_custom.Env):
    
    def __init__(self, host_ip, rate):
        self.host_ip = host_ip
        self.rate = ROSRate(rate)
        self.interface = URScriptInterface(host_ip)

        # UR3 (6DOF), 2F-85 gripper (1DOF)
        self._init_qpos = np.zeros([6])
        self._init_qvel = np.zeros([6])
        self._init_gripperpos = np.zeros([1])
        self._init_grippervel = np.zeros([1])

        self.action_space =  UR3RealEnv._set_action_space()
        obs = self._get_obs()
        self.observation_space = UR3RealEnv._set_observation_space(obs)

        self._episode_step = None

    def set_initial_joint_pos(self, q=None):
        if q is None: pass
        elif q == 'current': self._init_qpos = self.interface.get_joint_positions()
        else: self._init_qpos = q
        assert q.shape[0] == 6
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current': self._init_qvel = self.interface.get_joint_speeds()
        else: self._init_qvel = qd
        assert qd.shape[0] == 6
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current': self._init_gripperpos = self.interface.get_gripper_position()
        else: self._init_gripperpos = g
        assert g.shape[0] == 1
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current': self._init_grippervel = self.interface.get_gripper_position()
        else: self._init_grippervel = gd
        assert g.shape[0] == 1
        print('Initial gripper velocity is set to %s'%(gd))

    def step(self, action):
        start = time.time()
        assert self._episode_step is not None, 'Must reset before step!'
        for command_type, command_val in action.items():
            getattr(self.interface, command_type)(command_val)
        self._episode_step += 1
        self.rate.sleep()
        ob = self._get_obs()
        reward = 1.0
        done = False
        finish = time.time()
        if finish - start > 1.5*self.rate._actual_cycle_time:
            raise warnings.warn('Desired rate of %dHz is not satisfied! (current rate: %dHz)'%(self.rate._freq, 1/(finish-start)))
        return ob, reward, done, {}

    def reset(self):
        self.interface.reset_controller()
        ob = self.reset_model()
        self.rate.reset()
        return ob

    def render(self, mode='human'):
        warnings.warn('Real environment. "Render" with your own two eyes!')

    def close(self):
        self.interface.close()

    def reset_model(self):
        self.interface.movej(q=self._init_qpos)
        self.interface.move_gripper(q=self._init_gripperpos)
        self._episode_step = 0
        return self._get_obs()

    def _get_obs_dict(self):
        return {
            'qpos': self.interface.get_joint_positions(),
            'qvel': self.interface.get_joint_speeds(),
            'gripperpos': self.interface.get_gripper_position(),
            'grippervel': self.interface.get_gripper_speed()
        }

    def _get_obs(self):
        return UR3RealEnv._dict_to_nparray(self._get_obs_dict())

    @staticmethod
    def _dict_to_nparray(obs_dict):
        return np.array([obs_dict['qpos'], obs_dict['gripperpos'], obs_dict['qvel'], obs_dict['grippervel']])

    @staticmethod
    def _nparray_to_dict(obs_nparray):
        return {
            'qpos': obs_nparray[0:6],
            'qvel': obs_nparray[7:13],
            'gripperpos': obs_nparray[6:7],
            'grippervel': obs_nparray[13:14]
        }

    @staticmethod
    def _set_action_space():
        return convert_action_to_space(COMMAND_LIMITS)

    @staticmethod
    def _set_observation_space(observation):
        return convert_observation_to_space(observation)


if __name__ == "__main__":
    
    real_env = UR3RealEnv(host_ip='192.168.5.101', rate=100)
    obs = real_env.reset()
    init_qpos = obs['qpos']
    goal_qpos = init_qpos.copy()
    goal_qpos[-1] = np.pi/2
    waypoints_qpos = np.linspace(init_qpos, goal_qpos, 200, axis=0)
    waypoints_qvel = np.diff(waypoints_qpos, axis=0)/real_env.rate
    
    # open-close-open gripper
    real_env.step({'open_gripper': None})
    time.sleep(1.0)
    real_env.step({'close_gripper': None})
    time.sleep(1.0)
    real_env.step({'open_gripper': None})
    time.sleep(2.0)

    if prompt_yes_or_no('servoj to %s deg?'%(np.deg2rad(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qpos[1:,:]):
        action = {
            'servoj': waypoint,
            'open_gripper': None
        }
        real_env.step(action)
        print('action %d sent!'%(n))
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    print('Moving to initial position...')
    real_env.step({'movej': waypoints_qpos[0,:]})
    print('done!')

    if prompt_yes_or_no('servoj to %s deg?'%(np.deg2rad(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # speedj example
    print('Testing speedj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qvel):
        action = {
            'speedj': waypoint,
            'open_gripper': None
        }
        real_env.step(action)
        print('action %d sent!'%(n))
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    print('Moving to initial position...')
    real_env.step({'movej': waypoints_qpos[0,:]})
    print('done!')
    
    # close-open-close gripper
    real_env.step({'close_gripper': None})
    time.sleep(1.0)
    real_env.step({'open_gripper': None})
    time.sleep(1.0)
    real_env.step({'close_gripper': None})
    time.sleep(2.0)