import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, COMMAND_LIMITS


def convert_action_to_space(action_limits=COMMAND_LIMITS):
    if isinstance(action_limits, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in COMMAND_LIMITS.items()
        ]))
    elif isinstance(action_limits, list):
        low = action_limits[0]
        high = action_limits[1]
        space = gym_custom.spaces.Box(low, high, dtype=action_limits.dtype)
    else:
        raise NotImplementedError(type(action_limits), action_limits)

    return space

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class UR3RealEnv(gym_custom.Env):
    
    def __init__(self, host_ip, rate):
        self.host_ip = host_ip
        self.interface = URScriptInterface(host_ip)

        # UR3 (6DOF), 2F-85 gripper (1DOF)
        self._init_qpos = np.zeros([6])
        self._init_qvel = np.zeros([6])
        self._init_gripperpos = np.zeros([1])
        self._init_grippervel = np.zeros([1])

        self._set_action_space()
        obs = self._get_obs()
        self._set_observation_space(obs)

        self._episode_step = None

    def set_initial_joint_pos(self, q=None):
        if q is None: pass
        elif q == 'current': self._init_qpos = self.interface.get_joint_positions()
        else: self._init_qpos = q
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current': self._init_qvel = self.interface.get_joint_speeds()
        else: self._init_qvel = qd
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current': self._init_gripperpos = self.interface.get_gripper_position()
        else: self._init_gripperpos = g
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current': self._init_grippervel = self.interface.get_gripper_position()
        else: self._init_grippervel = gd
        print('Initial gripper velocity is set to %s'%(gd))

    def step(self, action):
        assert self._episode_step is not None, 'Must reset before step!'
        for command in action:
            getattr(self.interface, command['type'])(command['data'])

    def reset(self):
        self.interface.reset_controller()
        ob = self.reset_model()

    def render(self, mode='human'):
        warnings.warn('Real environment. "Render" with your own two eyes!')

    def close(self):
        self.interface.close()

    def reset_model(self):
        self.interface.movej(q=self._init_qpos)
        self.interface.move_gripper(q=self._init_gripperpos)
        self._episode_step = 0

    def _get_obs(self):
        return {
            'qpos': self.interface.get_joint_positions(),
            'qvel': self.interface.get_joint_speeds(),
            'gripperpos': self.interface.get_gripper_position(),
            'grippervel': self.interface.get_gripper_speed()
        }

    def _set_action_space(self):
        self.action_space = convert_action_to_space()
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space


# TODO:
if __name__ == "__main__":
    # simple example
    pass