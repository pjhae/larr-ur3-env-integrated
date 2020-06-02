import numpy as np
import os
from gym_custom import utils
from gym_custom.core import ActionWrapper
from gym_custom.envs.mujoco import MujocoEnv


class DualUR3Env(MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        xml_filename = 'dual_ur3_larr.xml'
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', 'ur3', xml_filename)
        MujocoEnv.__init__(self, fullpath, 1)

        self.ur3_nqpos, self.gripper_nqpos, self.objects_nqpos = 6, 10, [7]
        self.ur3_nqvel, self.gripper_nqvel, self.objects_nqvel = 6, 10, [6]
        assert 2*self.ur3_nqpos + 2*self.gripper_nqpos + sum(self.objects_nqpos) == self.model.nq, 'Number of qpos elements mismatch'
        assert 2*self.ur3_nqvel + 2*self.gripper_nqvel + sum(self.objects_nqvel) == self.model.nv, 'Number of qvel elements mismatch'
        self.ur3_nact, self.gripper_nact = 12, 4
        assert self.ur3_nact + self.gripper_nact == self.model.nu, 'Number of action elements mismatch'

        # Initial position for UR3
        self.init_qpos[0:self.ur3_nqpos] = \
            np.array([-90.0, -90.0, -90.0, -45.0, 225.0, 0.0])*np.pi/180.0 # right arm
        self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
            np.array([90.0, -90.0, 90.0, 225.0, 135.0, 0.0])*np.pi/180.0 # left arm
        # Initial postion for gripper
        self.init_qpos[8], self.init_qpos[13] = 0.04, 0.04

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

    def _get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos], 
            self.sim.data.qpos[self.ur3_nqpos+gripper_nqpos:2*self.ur3_nqpos+gripper_nqpos]]).ravel()

    def _get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos], 
            self.sim.data.qpos[2*self.ur3_nqpos+gripper_nqpos:2*self.ur3_nqpos+2*gripper_nqpos]]).ravel()

    def _get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel], 
            self.sim.data.qvel[self.ur3_nqvel+gripper_nqvel:2*self.ur3_nqvel+gripper_nqvel]]).ravel()

    def _get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qvel[2*self.ur3_nqvel+gripper_nqvel:2*self.ur3_nqvel+2*gripper_nqvel]]).ravel()

    def _get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel], 
            self.sim.data.qfrc_bias[self.ur3_nqvel+gripper_nqvel:2*self.ur3_nqvel+gripper_nqvel]]).ravel()

    def _get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel], 
            self.sim.data.qfrc_bias[self.ur3_nqvel+gripper_nqvel:2*self.ur3_nqvel+gripper_nqvel]]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

class URScriptWrapper(ActionWrapper):
    '''
    UR Script Wrapper for DualUR3Env
    '''
    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        
        self.ndof, self.ngripperdof = ur3_scale_factor.shape[0], gripper_scale_factor.shape[0]
        self.scale_factor_ur3 = np.concatenate([ur3_scale_factor, ur3_scale_factor])
        self.scale_factor_gripper = np.concatenate([gripper_scale_factor, gripper_scale_factor])
        assert self.ndof == self.env.ur3_nact/2 and self.ngripperdof == self.env.gripper_nact/2, 'DOF mismatch'
        
        self.PID_gains = copy.deepcopy(PID_gains)
        self.ur3_err_integ, self.gripper_err_integ = 0.0, 0.0
        # self.ur3_err_integ_limits, self.gripper_err_integ_limits = [-2.5, 2.5], [-1.0, 1.0]
        
        self.ur3_command_type, self.gripper_command_type = None, None

    def action(self, ur_command, relative=False):
        ur3_command_type_list = ['servoj', 'speedj']
        gripper_command_type_list = ['position', 'velocity', 'force']

        if ur3_command_type != ur_command['ur3']['type']:
            self.ur3_err_integ = 0.0 # Clear integration term after command type change
            self.ur3_command_type = ur_command['ur3']['type']
        if gripper_command_type != ur_command['gripper']['type']:
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
        if gripper_command['gripper']['type'] == gripper_command_type_list[0]:
            gripper_action = self._positiong(q=ur_command['gripper']['command'])
        elif ur_command['gripper']['type'] == gripper_command_type_list[1]:
            gripper_action = self._velocityg(qf=ur_command['gripper']['command'])
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
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -2.5, 2.5)

        # Internal forces
        bias = self.env._get_ur3_bias()

        # PID controller
        action = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ + self.PID_gains['D']*err_dot) + bias
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
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err*self.env.dt, -2.5, 2.5)

        # Internal forces
        bias = self.env._get_ur3_bias()

        # PID controller
        action = self.ur3_scale_factor*(self.PID_gains['P']*err + self.PID_gains['I']*self.ur3_err_integ) + bias
        return action

    def _positiong(self, q):
        assert q.shape[0] == self.ngripperdof
        bias = self.env._get_ur3_bias() # Internal forces
        # TODO:
        raise NotImplementedError
        return action
    
    def _velocityg(self, qd):
        assert q.shape[0] == self.ngripperdof
        bias = self.env._get_ur3_bias() # Internal forces
        # TODO:
        raise NotImplementedError
        return action

    def _forceg(self, qf):
        assert qf.shape[0] == self.ngripperdof
        bias = self.env._get_ur3_bias() # Internal forces
        action = np.array([qf[0], qf[0], qf[1], qf[1]]) + bias
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)
