import numpy as np
from collections import OrderedDict
import gym_custom
from gym_custom import spaces


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
            (key, convert_action_to_space(value))
            for key, value in COMMAND_LIMITS.items()
        ]))
    elif isinstance(action_limits, list):
        low = action_limits[0]
        high = action_limits[1]
        space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
    else:
        raise NotImplementedError(type(action_limits), action_limits)

    return space
    
    
def _set_action_space():
    return convert_action_to_space({'right': COMMAND_LIMITS})
    
    
 
 
print(_set_action_space()['speedj'].low)
