import copy
import numpy as np

from gym_custom.core import ActionWrapper


### Null objectiv base class(es)

class NullObjectiveBase(object):
    '''
    Base class for inverse kinematics null objective

    Must overload __init__() and _evaluate()
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, SO3):
        return self._evaluate(SO3)

    def _evaluate(self, SO3):
        raise NotImplementedError


### ActionWrapper class(es)

class URScriptWrapper_DualUR3_old(ActionWrapper):
    '''
    UR Script Wrapper for DualUR3Env
    '''
    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)
        self.ur3_scale_factor = np.concatenate([ur3_scale_factor, ur3_scale_factor])
        self.gripper_scale_factor = np.concatenate([gripper_scale_factor, gripper_scale_factor])
        self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
        assert self.ndof == self.env.ur3_nact/2 and self.ngripperdof == self.env.gripper_nact/2, 'DOF mismatch'
        
        self.PID_gains = copy.deepcopy(PID_gains)
        self.ur3_err_integ, self.gripper_err_integ = 0.0, 0.0
        # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
        
        self.ur3_command_type, self.gripper_command_type = None, None

        self.ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0, 50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

    def action(self, ur_command, relative=False):
        ur3_command_type_list = ['servoj', 'speedj']
        gripper_command_type_list = ['positiong', 'velocityg', 'forceg']

        if self.ur3_command_type != ur_command['ur3']['type']:
            self.ur3_err_integ = 0.0 # Clear integration term after command type change
            self.ur3_command_type = ur_command['ur3']['type']
        if self.gripper_command_type != ur_command['gripper']['type']:
            self.gripper_err_integ = 0.0
            self.gripper_command_type = ur_command['gripper']['type']

        # UR3 commands
        if ur_command['ur3']['type'] == ur3_command_type_list[0]:
            ur3_action = self._servoj(q=ur_command['ur3']['command'], a=None, v=None)
        elif ur_command['ur3']['type'] == ur3_command_type_list[1]:
            ur3_action = self._speedj(qd=ur_command['ur3']['command'], a=None)            
        else:
            raise ValueError('Invalid UR3 command type!')
        # gripper commands
        if ur_command['gripper']['type'] == gripper_command_type_list[0]:
            gripper_action = self._positiong(q=ur_command['gripper']['command'])
        elif ur_command['gripper']['type'] == gripper_command_type_list[1]:
            gripper_action = self._velocityg(qd=ur_command['gripper']['command'])
        elif ur_command['gripper']['type'] == gripper_command_type_list[2]:
            gripper_action = self._forceg(qf=ur_command['gripper']['command'])
        else:
            raise ValueError('Invalid gripper command type!')
        
        return np.concatenate([ur3_action, gripper_action])

    def _servoj(self, q, a, v, t=0.008, lookahead_time=0.1, gain=300):
        '''
        from URScript API Reference v3.5.4

            q: joint positions (rad)
            a: NOT used in current version
            v: NOT used in current version
            t: time where the command is controlling the robot. The function is blocking for time t [S]
            lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
            gain: proportional gain for following target position, range [100,2000]
        '''
        assert q.shape[0] == 2*self.ndof
        # Calculate error
        current_theta = self.env._get_ur3_qpos()
        # if ur3_command['relative']: # Relative position
        #     theta_dist = np.mod(ur3_command['desired'] - current_theta, 2*np.pi)
        #     err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        # else: # Absolute position
        #     err = ur3_command['desired'] - current_theta
        err = q - current_theta
        err_dot = -self.env._get_ur3_qvel()
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -1, 1)

        # Internal forces
        bias = self.env._get_ur3_bias()

        # External forces
        constraint = self.env._get_ur3_constraint()
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

        # PID controller
        # control_budget_high = self.ur3_torque_limit - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PID_control = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ + self.PID_gains['D']*err_dot)

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PID_control + bias - constraint
        return action

    def _speedj(self, qd, a, t=None):
        '''
        from URScript API Reference v3.5.4
            qd: joint speeds (rad/s)
            a: joint acceleration [rad/s^2] (of leading axis)
            t: time [s] before the function returns (optional)
        '''
        assert qd.shape[0] == 2*self.ndof
        # Calculate error
        current_thetadot = self.env._get_ur3_qvel()
        err = qd - current_thetadot
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -0.02, 0.02)

        # Internal forces
        bias = self.env._get_ur3_bias()

        # External forces
        constraint = self.env._get_ur3_constraint()
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit, 0.50*self.ur3_torque_limit)

        # PID controller
        # control_budget_high = self.ur3_torque_limit - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PI_control = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ)

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PI_control + bias - constraint
        return action

    def _positiong(self, q):
        assert q.shape[0] == self.ngripperdof
        bias = self.env._get_gripper_bias() # Internal forces
        err = np.array([q[0], q[0], q[1], q[1]]) - self.env._get_gripper_qpos()
        action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        return action
    
    def _velocityg(self, qd):
        assert qd.shape[0] == self.ngripperdof
        bias = self.env._get_gripper_bias() # Internal forces
        err = np.array([qd[0], qd[0], qd[1], qd[1]]) - self.env._get_gripper_qvel()
        action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        return action

    def _forceg(self, qf):
        assert qf.shape[0] == self.ngripperdof
        bias = self.env._get_gripper_bias() # Internal forces
        action = np.array([qf[0], qf[0], qf[1], qf[1]]) + np.array([bias[2], bias[7], bias[12], bias[17]])
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)


class URScriptWrapper_DualUR3_new(ActionWrapper):
    '''
    UR Script Wrapper for DualUR3Env
    '''
    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)
        self.ur3_scale_factor = np.concatenate([ur3_scale_factor, ur3_scale_factor]) # 2 x ndof
        self.gripper_scale_factor = np.concatenate([gripper_scale_factor, gripper_scale_factor]) # 2 x ngripper
        self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
        assert self.ndof == self.env.ur3_nact and self.ngripperdof == self.env.gripper_nact, 'DOF mismatch'
        
        self.servoj_gains = PID_gains['servoj']
        self.speedj_gains = PID_gains['speedj']
        self.ur3_err_integ, self.gripper_err_integ = np.zeros([2*self.ndof]), np.zeros([2*self.ngripperdof])
        # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
        
        self.ur3_right_command_type, self.ur3_left_command_type = None, None
        self.gripper_right_command_type, self.gripper_left_command_type = None, None

        self.ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0, 50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

    def action(self, ur_command, relative=False):
        ur3_command_type_list = ['servoj', 'speedj', 'none']
        gripper_command_type_list = ['positiong', 'velocityg', 'forceg', 'none']

        # Clear integration term after command type change
        if self.ur3_right_command_type != ur_command['ur3_right']['type']:
            self.ur3_err_integ[:self.ndof] = np.zeros([self.ndof])
            self.ur3_right_command_type = ur_command['ur3_right']['type']
        if self.ur3_left_command_type != ur_command['ur3_left']['type']:
            self.ur3_err_integ[self.ndof:] = np.zeros([self.ndof])
            self.ur3_left_command_type = ur_command['ur3_left']['type']
        if self.gripper_right_command_type != ur_command['gripper_right']['type']:
            self.gripper_err_integ[:self.ngripperdof] = np.zeros([self.ngripperdof])
            self.gripper_right_command_type = ur_command['gripper_right']['type']
        if self.gripper_left_command_type != ur_command['gripper_left']['type']:
            self.gripper_err_integ[self.ngripperdof:] = np.zeros([self.ngripperdof])
            self.gripper_left_command_type = ur_command['gripper_left']['type']

        # UR3 (right) commands
        if ur_command['ur3_right']['type'] == ur3_command_type_list[0]:
            ur3_right_action = self._servoj(q=ur_command['ur3_right']['command'], a=None, v=None, idxes=[0,self.ndof])
        elif ur_command['ur3_right']['type'] == ur3_command_type_list[1]:
            ur3_right_action = self._speedj(qd=ur_command['ur3_right']['command'], a=None, idxes=[0,self.ndof])
        elif ur_command['ur3_right']['type'] == ur3_command_type_list[2]:
            ur3_right_action = np.zeros([self.ndof])
        else:
            raise ValueError('Invalid UR3 command type!')
        # UR3 (left) commands
        if ur_command['ur3_left']['type'] == ur3_command_type_list[0]:
            ur3_left_action = self._servoj(q=ur_command['ur3_left']['command'], a=None, v=None, idxes=[self.ndof,2*self.ndof])
        elif ur_command['ur3_left']['type'] == ur3_command_type_list[1]:
            ur3_left_action = self._speedj(qd=ur_command['ur3_left']['command'], a=None, idxes=[self.ndof,2*self.ndof])
        elif ur_command['ur3_left']['type'] == ur3_command_type_list[2]:
            ur3_left_action = np.zeros([self.ndof])
        else:
            raise ValueError('Invalid UR3 command type!')
        # gripper (right) commands
        if ur_command['gripper_right']['type'] == gripper_command_type_list[0]:
            gripper_right_action = self._positiong(q=ur_command['gripper_right']['command'], idx=0)
        elif ur_command['gripper_right']['type'] == gripper_command_type_list[1]:
            gripper_right_action = self._velocityg(qd=ur_command['gripper_right']['command'], idx=0)
        elif ur_command['gripper_right']['type'] == gripper_command_type_list[2]:
            gripper_right_action = self._forceg(qf=ur_command['gripper_right']['command'], idx=0)
        elif ur_command['gripper_right']['type'] == gripper_command_type_list[3]:
            gripper_right_action = np.zeros([self.ngripperdof])
        else:
            raise ValueError('Invalid gripper command type!')
        # gripper (left) commands
        if ur_command['gripper_left']['type'] == gripper_command_type_list[0]:
            gripper_left_action = self._positiong(q=ur_command['gripper_left']['command'], idx=1)
        elif ur_command['gripper_left']['type'] == gripper_command_type_list[1]:
            gripper_left_action = self._velocityg(qd=ur_command['gripper_left']['command'], idx=1)
        elif ur_command['gripper_left']['type'] == gripper_command_type_list[2]:
            gripper_left_action = self._forceg(qf=ur_command['gripper_left']['command'], idx=1)
        elif ur_command['gripper_left']['type'] == gripper_command_type_list[3]:
            gripper_left_action = np.zeros([self.ngripperdof])
        else:
            raise ValueError('Invalid gripper command type!')
        
        return np.concatenate([ur3_right_action, ur3_left_action, gripper_right_action, gripper_left_action])

    def _servoj(self, q, a, v, idxes, t=0.008, lookahead_time=0.1, gain=300):
        '''
        from URScript API Reference v3.5.4

            q: joint positions (rad)
            a: NOT used in current version
            v: NOT used in current version
            t: time where the command is controlling the robot. The function is blocking for time t [S]
            lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
            gain: proportional gain for following target position, range [100,2000]
        '''
        assert q.shape[0] == self.ndof
        # Calculate error
        current_theta = self.env._get_ur3_qpos()[idxes[0]:idxes[1]]
        # if ur3_command['relative']: # Relative position
        #     theta_dist = np.mod(ur3_command['desired'] - current_theta, 2*np.pi)
        #     err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        # else: # Absolute position
        #     err = ur3_command['desired'] - current_theta
        err = q - current_theta
        err_dot = -self.env._get_ur3_qvel()[idxes[0]:idxes[1]]
        self.ur3_err_integ[idxes[0]:idxes[1]] = np.clip(self.ur3_err_integ[idxes[0]:idxes[1]] + err*self.env.dt, -1, 1)

        # Internal forces
        bias = self.env._get_ur3_bias()[idxes[0]:idxes[1]]

        # External forces
        constraint = self.env._get_ur3_constraint()[idxes[0]:idxes[1]]
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit[idxes[0]:idxes[1]], 0.50*self.ur3_torque_limit[idxes[0]:idxes[1]])

        # PID controller
        # control_budget_high = self.ur3_torque_limit[idxes[0]:idxes[1]] - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit[idxes[0]:idxes[1]] - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PID_control = self.ur3_scale_factor[idxes[0]:idxes[1]]*\
            (self.servoj_gains['P']*err + self.servoj_gains['I']*self.ur3_err_integ[idxes[0]:idxes[1]] + self.servoj_gains['D']*err_dot)

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PID_control + bias - constraint
        return action

    def _speedj(self, qd, a, idxes, t=None):
        '''
        from URScript API Reference v3.5.4
            qd: joint speeds (rad/s)
            a: joint acceleration [rad/s^2] (of leading axis)
            t: time [s] before the function returns (optional)
        '''
        assert qd.shape[0] == self.ndof
        # Calculate error
        current_thetadot = self.env._get_ur3_qvel()[idxes[0]:idxes[1]]
        err = qd - current_thetadot
        self.ur3_err_integ[idxes[0]:idxes[1]] = np.clip(self.ur3_err_integ[idxes[0]:idxes[1]] + err*self.env.dt, -0.02, 0.02)

        # Internal forces
        bias = self.env._get_ur3_bias()[idxes[0]:idxes[1]]

        # External forces
        constraint = self.env._get_ur3_constraint()[idxes[0]:idxes[1]]
        constraint = np.clip(constraint, -0.50*self.ur3_torque_limit[idxes[0]:idxes[1]], 0.50*self.ur3_torque_limit[idxes[0]:idxes[1]])

        # PID controller
        # control_budget_high = self.ur3_torque_limit[idxes[0]:idxes[1]] - (bias - constraint)
        # control_budget_high = np.maximum(control_budget_high, 0)
        # control_budget_low = -self.ur3_torque_limit[idxes[0]:idxes[1]] - (bias - constraint)
        # control_budget_low = np.minimum(control_budget_low, 0)

        PI_control = self.ur3_scale_factor[idxes[0]:idxes[1]]*\
            (self.speedj_gains['P']*err + self.speedj_gains['I']*self.ur3_err_integ[idxes[0]:idxes[1]])

        # scale_upper = np.min(np.where(PID_control > 0, control_budget_high/PID_control, np.inf))
        # scale_lower = np.min(np.where(PID_control < 0, control_budget_high/PID_control, np.inf))
        # rescale = min(scale_lower, scale_upper, 1)
        rescale = 1

        action = rescale*PI_control + bias - constraint
        return action

    def _positiong(self, q, idx):
        assert q.shape[0] == self.ngripperdof/2
        bias = self.env._get_gripper_bias() # Internal forces
        # err = np.array([q[0], q[0], q[1], q[1]]) - self.env._get_gripper_qpos()
        # action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        err = np.array([q[idx], q[idx]]) - self.env._get_gripper_qpos()[2*idx:2*idx+2]
        action = self.gripper_scale_factor[2*idx:2*idx+2]*err + np.array([bias[10*idx+2], bias[10*idx+7]]) # P control
        return action
    
    def _velocityg(self, qd, idx):
        assert qd.shape[0] == self.ngripperdof/2
        bias = self.env._get_gripper_bias() # Internal forces
        # err = np.array([qd[0], qd[0], qd[1], qd[1]]) - self.env._get_gripper_qvel()
        # action = self.gripper_scale_factor*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        err = np.array([qd[idx], qd[idx]]) - self.env._get_gripper_qvel()[2*idx:2*idx+2]
        action = self.gripper_scale_factor[2*idx:2*idx+2]*err + np.array([bias[10*idx+2], bias[10*idx+7]]) # P control
        return action

    def _forceg(self, qf, idx):
        assert qf.shape[0] == self.ngripperdof/2
        bias = self.env._get_gripper_bias() # Internal forces
        # action = np.array([qf[0], qf[0], qf[1], qf[1]]) + np.array([bias[2], bias[7], bias[12], bias[17]])
        action = np.array([qf[idx], qf[idx]]) + np.array([bias[10*idx+2], bias[10*idx+7]])
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)