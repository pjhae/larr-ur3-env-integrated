import time
import warnings

import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, convert_action_to_space, convert_observation_to_space, COMMAND_LIMITS
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

class DualUR3RealEnv(gym_custom.Env):
    
    def __init__(self, host_ip_right, host_ip_left, rate):
        self.host_ip_right = host_ip_right
        self.host_ip_left = host_ip_left
        self.rate = ROSRate(rate)
        self.interface_right = URScriptInterface(host_ip_right) 
        self.interface_left = URScriptInterface(host_ip_left)

        self._init_qpos = np.zeros([12])
        self._init_qvel = np.zeros([12])
        self._init_gripperpos = np.zeros([2])
        self._init_grippervel = np.zeros([2])

        self.action_space = DualUR3RealEnv._set_action_space()
        obs = self._get_obs()
        self.observation_space = DualUR3RealEnv._set_observation_space()

        self._episode_step = None

    def set_initial_joint_pos(self, q=None):
        if q is None: pass
        elif q == 'current': self._init_qpos = self.interface.get_joint_positions()
        else: self._init_qpos = q
        assert q.shape[0] == 12
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current': self._init_qvel = self.interface.get_joint_speeds()
        else: self._init_qvel = qd
        assert qd.shape[0] == 12
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current': self._init_gripperpos = self.interface.get_gripper_position()
        else: self._init_gripperpos = g
        assert g.shape[0] == 2
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current': self._init_grippervel = self.interface.get_gripper_position()
        else: self._init_grippervel = gd
        assert gd.shape[0] == 2
        print('Initial gripper velocity is set to %s'%(gd))

    def step(self, action):
        start = time.time()
        assert self._episode_step is not None, 'Must reset before step!'
        # TODO: Send commands to both arms simultaneously?
        for command_type, command_val in action['right'].items():
            getattr(self.interface_right, command_type)(command_val)
        for command_type, command_val in action['left'].items():
            getattr(self.interface_left, command_type)(command_val)
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
        # TODO: Send commands to both arms simultaneously?
        self.interface_right.reset_controller()
        self.interface_left.reset_controller()
        ob = self.reset_model()
        self.rate.reset()
        return ob

    def render(self, mode='human'):
        warnings.warn('Real environment. "Render" with your own two eyes!')

    def close(self):
        self.interface_right.close()
        self.interface_left.close()

    def reset_model(self):
        # TODO: Send commands to both arms simultaneously?
        self.interface_right.movej(q=self._init_qpos[:6])
        self.interface_left.movej(q=self._init_qpos[6:])
        self.interface_right.move_gripper(q=self._init_gripperpos[:1])
        self.interface_left.move_gripper(q=self._init_gripperpos[1:])
        self._episode_step = 0
        return self._get_obs()

    def _get_obs_dict(self):
        return {'right': {
                'qpos': self.interface_right.get_joint_positions(),
                'qvel': self.interface_right.get_joint_speeds(),
                'gripperpos': self.interface_right.get_gripper_position(),
                'grippervel': self.interface_right.get_gripper_speed()
            },
            'left': {
                'qpos': self.interface_left.get_joint_positions(),
                'qvel': self.interface_left.get_joint_speeds(),
                'gripperpos': self.interface_left.get_gripper_position(),
                'grippervel': self.interface_left.get_gripper_speed()
            }
        }

    def _get_obs(self):
        return DualUR3RealEnv._dict_to_nparray(self._get_obs_dict())

    @staticmethod
    def _dict_to_nparray(obs_dict):
        right = obs_dict['right']
        left = obs_dict['left']
        return np.array([right['qpos'], right['gripperpos'], left['qpos'], left['gripperpos'],
            right['qvel'], right['grippervel'], left['qvel'], left['grippervel']])

    @staticmethod
    def _nparray_to_dict(obs_nparray):
        return {'right': {
                'qpos': obs_nparray[0:6],
                'qvel': obs_nparray[14:20],
                'gripperpos': obs_nparray[6:7],
                'grippervel': obs_nparray[20:21]
            },
            'left': {
                'qpos': obs_nparray[7:13],
                'qvel': obs_nparray[21:27],
                'gripperpos': obs_nparray[13:14],
                'grippervel': obs_nparray[27:28]
            }
        }

    @staticmethod
    def _set_action_space():
        return convert_action_to_space({'right': COMMAND_LIMITS, 'left': COMMAND_LIMITS})

    @staticmethod
    def _set_observation_space(observation):
        return convert_observation_to_space(observation)

if __name__ == "__main__":
    
    real_env = DualUR3RealEnv(host_ip_right='192.168.5.102', host_ip_left='192.168.5.101', rate=100)
    obs = real_env.reset()
    init_qpos = obs['qpos']
    goal_qpos = init_qpos.copy()
    goal_qpos[-1] = np.pi/2
    waypoints_qpos = np.linspace(init_qpos, goal_qpos, 200, axis=0)
    waypoints_qvel = np.diff(waypoints_qpos, axis=0)/real_env.rate
    
    # open-close-open gripper
    real_env.step({'right': {'open_gripper': None}, 'left': {'open_gripper': None}})
    time.sleep(1.0)
    real_env.step({'right': {'close_gripper': None}, 'left': {'close_gripper': None}})
    time.sleep(1.0)
    real_env.step({'right': {'open_gripper': None}, 'left': {'open_gripper': None}})
    time.sleep(2.0)

    if prompt_yes_or_no('servoj to %s deg?'%(np.deg2rad(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qpos[1:,:]):
        action = {}
        action['right'] = {
            'servoj': waypoint,
            'open_gripper': None
        }
        action['left'] = {
            'servoj': waypoint,
            'open_gripper': None
        }
        real_env.step(action)
        print('action %d sent!'%(n))
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    print('Moving to initial position...')
    real_env.step({'right': {'movej': waypoints_qpos[0,:]}, 'left': {'movej': waypoints_qpos[0,:]}})
    print('done!')

    if prompt_yes_or_no('servoj to %s deg?'%(np.deg2rad(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # speedj example
    print('Testing speedj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qvel):
        action = {}
        action['right'] = {
            'speedj': waypoint,
            'open_gripper': None
        }
        action['right'] = {
            'left': waypoint,
            'open_gripper': None
        }
        real_env.step(action)
        print('action %d sent!'%(n))
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    print('Moving to initial position...')
    real_env.step({'right': {'movej': waypoints_qpos[0,:]}, 'left': {'movej': waypoints_qpos[0,:]}})
    print('done!')
    
    # close-open-close gripper
    real_env.step({'right': {'close_gripper': None}, 'left': {'close_gripper': None}})
    time.sleep(1.0)
    real_env.step({'right': {'open_gripper': None}, 'left': {'open_gripper': None}})
    time.sleep(1.0)
    real_env.step({'right': {'close_gripper': None}, 'left': {'close_gripper': None}})
    time.sleep(2.0)