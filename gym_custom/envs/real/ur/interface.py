import numpy as np

import gym_custom.envs.real.ur.drivers.URBasic


COMMAND_LIMITS = {
    'movej': [np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -np.inf]),
        np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, np.inf])], # [rad]
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])], # [rad/s]
    'move_gripper': [np.array([0]), np.array([1])] # [0: open, 1: close]
}

def convert_action_to_space(action_limits):
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
        space = gym_custom.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class URScriptInterface(object):
    
    def __init__(self, host_ip):
        
        gripper_kwargs = {
            'robot': None,
            'payload': 0.85,
            'speed': 255, # 0~255
            'force': 255,  # 0~255
            'socket_host': host_ip,
            'socket_port': find_free_port(),
            'socket_name': 'gripper_socket'
        }

        self.model = URBasic.robotModel.RobotModel()
        self.comm = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=self.model, **gripper_kwargs)

    def close(self):
        self.comm.close()

    ## UR Controller
    def reset_controller(self):
        self.comm.reset_error()

    ## UR3 manipulator
    def movej(self, *args, **kwargs):
        '''
        Move to position (linear in joint-space)
        blocking command, not suitable for online control
        '''
        self.comm.movej(*args, **kwargs)

    def movel(self, *args, **kwargs):
        raise NotImplementedError()

    def movep(self, *args, **kwargs):
        raise NotImplementedError()

    def movec(self, *args, **kwargs):
        raise NotImplementedError()
    
    def servoc(self, *args, **kwargs):
        raise NotImplementedError()

    def servoj(self, *args, **kwargs):
        '''
        Servo to position (linear in joint-space)
        non-blocking command, suitable for online control
        '''
        self.comm.servoj(*args, **kwargs)

    def speedj(self, *args, **kwargs):
        '''
        non-blocking command, suitable for online control
        '''
        self.comm.speedj(*args, **kwargs)

    def speedl(self, *args, **kwargs):
        raise NotImplementedError()
    
    def stopj(self, *args, **kwargs):
        '''
        ?
        '''
        self.comm.stopj(*args, **kwargs)

    def stopl(self, *args, **kwargs):
        raise NotImplementedError()

    def get_joint_positions(self):
        return np.array(self.comm.get_actual_joint_positions())

    def get_joint_speeds(self):
        return np.array(self.comm.get_actual_joint_speeds())

    ## 2F-85 gripper
    def open_gripper(self, *args, **kwargs):
        self.comm.operate_gripper(0)

    def close_gripper(self, *args, **kwargs):
        self.comm.operate_gripper(255)

    def move_gripper(self, *args, **kwargs):
        # TODO: dscho
        raise NotImplementedError()

    def get_gripper_position(self):
        # TODO: dscho
        raise NotImplementedError()

    def get_gripper_speed(self):
        # TODO: dscho
        raise NotImplementedError()