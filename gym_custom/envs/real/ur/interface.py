import numpy as np

import gym_custom.envs.real.ur.drivers.URBasic


COMMAND_LIMITS = {
    'movej': [np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -np.inf]),
        np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, np.inf])], # [rad]
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])], # [rad/s]
}

class URScriptInterface(object):
    
    def __init__(self, host_ip):
        self.model = URBasic.robotModel.RobotModel()
        self.comm = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=self.model)

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
        # TODO: dscho
        raise NotImplementedError()

    def close_gripper(self, *args, **kwargs):
        # TODO: dscho
        raise NotImplementedError()

    def move_gripper(self, *args, **kwargs):
        # TODO: dscho
        raise NotImplementedError()

    def get_gripper_position(self):
        # TODO: dscho
        raise NotImplementedError()

    def get_gripper_speed(self):
        # TODO: dscho
        raise NotImplementedError()