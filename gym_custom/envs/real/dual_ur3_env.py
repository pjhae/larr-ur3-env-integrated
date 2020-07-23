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
        elif q == 'current':
            self._init_qpos = np.concatenate([self.interface_right.get_joint_positions(), self.interface_left.get_joint_positions()]).ravel()
        else:
            assert q.shape[0] == 12
            self._init_qpos = q
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current':
            self._init_qvel = np.concatenate([self.interface_right.get_joint_speeds(), self.interface_left.get_joint_speeds()]).ravel()
        else:
            assert qd.shape[0] == 12
            self._init_qvel = qd
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current':
            self._init_gripperpos = np.concatenate([self.interface_right.get_gripper_position(), self.interface_left.get_gripper_position()]).ravel()
        else:
            assert g.shape[0] == 2
            self._init_gripperpos = g
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current':
            self._init_gripperpos = np.concatenate([self.interface_right.get_gripper_speed(), self.interface_left.get_gripper_speed()]).ravel()
        else:
            assert gd.shape[0] == 2
            self._init_grippervel = gd
        print('Initial gripper velocity is set to %s'%(gd))

    def step(self, action):
        start = time.time()
        assert self._episode_step is not None, 'Must reset before step!'
        # TODO: Send commands to both arms simultaneously?
        for command_type, command_val in action['right'].items():
            getattr(self.interface_right, command_type)(**command_val)
        for command_type, command_val in action['left'].items():
            getattr(self.interface_left, command_type)(**command_val)
        self._episode_step += 1
        self.rate.sleep()
        ob = self._get_obs()
        reward = 1.0
        done = False
        finish = time.time()
        if finish - start > 1.5/self.rate._freq:
            warnings.warn('Desired rate of %dHz is not satisfied! (current rate: %dHz)'%(self.rate._freq, 1/(finish-start)))
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
        return np.concatenate([right['qpos'], right['gripperpos'], left['qpos'], left['gripperpos'],
            right['qvel'], right['grippervel'], left['qvel'], left['grippervel']]).ravel()

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


## Examples
def servoj_speedj_example(host_ip_right, host_ip_left, rate):
    real_env = DualUR3RealEnv(host_ip_right=host_ip_right, host_ip_left=host_ip_left, rate=rate)
    real_env.set_initial_joint_pos('current')
    real_env.set_initial_gripper_pos('current')
    if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
        %(np.rad2deg(real_env._init_qpos[:6]), np.rad2deg(real_env._init_qpos[6:]))) is False:
        print('exiting program!')
        sys.exit()
    obs = real_env.reset()
    obs_dict = real_env._nparray_to_dict(obs)
    init_qpos_right, init_qpos_left = obs_dict['right']['qpos'], obs_dict['left']['qpos']
    goal_qpos_right, goal_qpos_left = init_qpos_right.copy(), init_qpos_left.copy()
    goal_qpos_right[-1] += np.pi/2*1.5
    goal_qpos_left[-1] += np.pi/2*1.5
    waypoints_qpos_right = np.linspace(init_qpos_right, goal_qpos_right, rate*2, axis=0)
    waypoints_qpos_left = np.linspace(init_qpos_left, goal_qpos_left, rate*2, axis=0)
    waypoints_qvel_right = np.diff(waypoints_qpos_right, axis=0)*real_env.rate._freq
    waypoints_qvel_left = np.diff(waypoints_qpos_left, axis=0)*real_env.rate._freq
    
    # close-open-close gripper
    print('close')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(3.0)
    print('open')
    real_env.step({'right': {'open_gripper': {}}, 'left': {'open_gripper': {}}})
    time.sleep(3.0)
    print('close')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(5.0)

    if prompt_yes_or_no('servoj to \r\n right: %s deg\r\n left: %s deg\r\n?'
        %(np.rad2deg(goal_qpos_right), np.rad2deg(goal_pos_left))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, (waypoint_right, waypoint_left) in enumerate(zip(waypoints_qpos_right[1:,:], waypoints_qpos_left[1:,:])):
        real_env.step({
            'right': {
                'servoj': {'q': waypoint_right, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            },
            'left': {
                'servoj': {'q': waypoint_left, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            }
        })
        print('action %d sent!'%(n))
    real_env.step({'right': {'stopj': {'a': 5}}, 'left': {'stopj': {'a': 5}}})
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_obs_dict = real_env._nparray_to_dict(real_env._get_obs())
    curr_qpos_right, curr_qpos_left = curr_obs_dict['right']['qpos'], curr_obs_dict['left']['qpos']
    print('current - goal qpos is \r\n right: %s deg\r\n left: %s deg'
        %(np.rad2deg(current_qpos_right - goal_qpos_right), np.rad2deg(current_qpos_left - goal_qpos_left)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'right': {'movej': {'q': waypoints_qpos_right[0,:]}}, 'left': {'movej': {'q': waypoints_qpos_left[0,:]}}})
    print('done!')

    if prompt_yes_or_no('speedj to \r\n right: %s deg\r\n left: %s deg\r\n?'
        %(np.rad2deg(goal_qpos_right), np.rad2deg(goal_pos_left))) is False:
        print('exiting program!')
        sys.exit()
    # speedj example
    print('Testing speedj')
    start = time.time()
    for n, (waypoint_right, waypoint_left) in enumerate(zip(waypoints_qvel_right, waypoints_qvel_left)):
        real_env.step({
            'right': {
                'speedj': {'qd': waypoint_right, 'a': 5, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            },
            'left': {
                'speedj': {'qd': waypoint_left, 'a': 5, 't': 2/real_env.rate._freq, 'wait': False},
                # 'open_gripper': {}
            }
        })
        print('action %d sent!'%(n))
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_obs_dict = real_env._nparray_to_dict(real_env._get_obs())
    curr_qpos_right, curr_qpos_left = curr_obs_dict['right']['qpos'], curr_obs_dict['left']['qpos']
    print('current - goal qpos is \r\n right: %s deg\r\n left: %s deg'
        %(np.rad2deg(current_qpos_right - goal_qpos_right), np.rad2deg(current_qpos_left - goal_qpos_left)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'right': {'movej': {'q': waypoints_qpos_right[0,:]}}, 'left': {'movej': {'q': waypoints_qpos_left[0,:]}}})
    print('done!')
    
    # open-close-open gripper
    print('open')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(3.0)
    print('close')
    real_env.step({'right': {'open_gripper': {}}, 'left': {'open_gripper': {}}})
    time.sleep(3.0)
    print('open')
    real_env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    time.sleep(5.0)

if __name__ == "__main__":
    # servoj_speedj_example()
    pass