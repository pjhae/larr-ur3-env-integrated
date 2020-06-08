import copy
import numpy as np
import os
import warnings

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
            np.array([-90.0, -90.0, -90.0, -90.0, -135.0, 90.0])*np.pi/180.0 # right arm
        self.init_qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos] = \
            np.array([90.0, -90.0, 90.0, -90.0, 135.0, -90.0])*np.pi/180.0 # left arm
        
        # Settings for forward/inverse kinematics
        # https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
        self.kinematics_params = {}
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12*1]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])
        
        self.kinematics_params['T_wb_right'] = np.eye(4)
        right_base_idx = self.model.body_names.index('right_arm_rotz')
        self.kinematics_params['T_wb_right'][0:3,0:3] = self.sim.data.body_xmat[right_base_idx].reshape([3,3])
        self.kinematics_params['T_wb_right'][0:3,3] = self.sim.data.body_xpos[right_base_idx]
        
        self.kinematics_params['T_wb_left'] = np.eye(4)
        left_base_idx = self.model.body_names.index('left_arm_rotz')
        self.kinematics_params['T_wb_left'][0:3,0:3] = self.sim.data.body_xmat[left_base_idx].reshape([3,3])
        self.kinematics_params['T_wb_left'][0:3,3] = self.sim.data.body_xpos[left_base_idx]

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

    def forward_kinematics_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos

        if arm == 'right':
            T_0_i = self.kinematics_params['T_wb_right']
        elif arm == 'left':
            T_0_i = self.kinematics_params['T_wb_left']
        else:
            raise ValueError('Invalid arm type!')
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])
        # Base frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        for i in range(self.ur3_nqpos):
            ct = np.cos(q[i] + self.kinematics_params['offset'][i])
            st = np.sin(q[i] + self.kinematics_params['offset'][i])
            ca = np.cos(self.kinematics_params['alpha'][i])
            sa = np.sin(self.kinematics_params['alpha'][i])

            T_i_iplus1 = np.array([[ct, -st*ca, st*sa, self.kinematics_params['a'][i]*ct],
                                   [st, ct*ca, -ct*sa, self.kinematics_params['a'][i]*st],
                                   [0, sa, ca, self.kinematics_params['d'][i]],
                                   [0, 0, 0, 1]])
            T_0_i = np.matmul(T_0_i, T_i_iplus1)
            # cf. base frame at i=0
            T[i+1, :, :] = T_0_i
            R[i+1, :, :] = T_0_i[0:3,0:3]
            p[i+1, :] = T_0_i[0:3,3]

        return R, p, T

    def forward_kinematics_ee(self, q, arm):
        R, p, T = self.forward_kinematics_DH(q, arm)
        return R[-1,:,:], p[-1,:], T[-1,:,:]

    def _jacobian_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        _, ps, _ = self.forward_kinematics_DH(q, arm)
        p = ps[-1,:] # unperturbed position

        jac = np.zeros([3, self.ur3_nqpos])
        for i in range(self.ur3_nqpos):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            _, ps_, _ = self.forward_kinematics_DH(q_, arm)
            p_ = ps_[-1,:] # perturbed position
            jac[:, i] = (p_ - p)*epsilon_inv

        return jac

    def inverse_kinematics_ee(self, ee_pos, null_obj_func, arm,
            q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
        ):
        '''
        inverse kinematics with forward_kinematics_DH() and _jacobian_DH()
        '''
        # Set initial guess
        if arm == 'right':
            if q_init == 'current': q = self._get_ur3_qpos()[:self.ur3_nqpos]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            elif type(q_init).__name__ == 'ndarray': q = q_init
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':
            if q_init == 'current': q = self._get_ur3_qpos()[self.ur3_nqpos:]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            elif type(q_init).__name__ == 'ndarray': q = q_init
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        else:
            raise ValueError('Invalid arm type!')

        arm_to_body_name = {'right': 'right_gripper:hand', 'left': 'left_gripper:hand'}
        
        SO3, x, _ = self.forward_kinematics_ee(q, arm)
        jac = self._jacobian_DH(q, arm)
        delta_x = ee_pos - x
        err = np.linalg.norm(delta_x)
        null_obj_val = null_obj_func.evaluate(SO3)
        iter_taken = 0

        while True:
            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach
            jac_dagger = np.linalg.pinv(jac)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac) # null space of Jacobian
            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)
            for i in range(self.ur3_nqpos):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb, arm)
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)
                phi[i] = (null_obj_val_perturb - null_obj_val)/epsilon
            # update
            delta_x = ee_pos - x
            delta_q = np.matmul(jac_dagger, delta_x) - np.matmul(jac_null, phi)
            q += delta_q
            q = np.minimum(self.kinematics_params['ub'], np.maximum(q, self.kinematics_params['lb'])) # clip within theta bounds
            SO3, x, _ = self.forward_kinematics_ee(q, arm)
            jac = self._jacobian_DH(q, arm)
            null_obj_val = null_obj_func.evaluate(SO3)
            # evaluate
            err = np.linalg.norm(delta_x)
        
        if iter_taken == max_iter:
            warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null),
                RuntimeWarning)
        
        return q, iter_taken, err, null_obj_val

    def get_body_se3(self, body_name):
        # assert len(q) == self.ur3_nqpos
        body_idx = self.model.body_names.index(body_name)
        R = self.sim.data.body_xmat[body_idx].reshape([3,3])
        p = self.sim.data.body_xpos[body_idx]
        T = np.eye(4)
        T[0:3,0:3] = R
        T[0:3,3] = p

        return R, p, T

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def _get_ur3_qpos(self):
        return np.concatenate([self.sim.data.qpos[0:self.ur3_nqpos], 
            self.sim.data.qpos[self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+self.gripper_nqpos]]).ravel()

    def _get_gripper_qpos(self):
        return np.concatenate([self.sim.data.qpos[self.ur3_nqpos:self.ur3_nqpos+self.gripper_nqpos], 
            self.sim.data.qpos[2*self.ur3_nqpos+self.gripper_nqpos:2*self.ur3_nqpos+2*self.gripper_nqpos]]).ravel()

    def _get_ur3_qvel(self):
        return np.concatenate([self.sim.data.qvel[0:self.ur3_nqvel], 
            self.sim.data.qvel[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_gripper_qvel(self):
        return np.concatenate([self.sim.data.qvel[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qvel[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def _get_ur3_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[0:self.ur3_nqvel], 
            self.sim.data.qfrc_bias[self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+self.gripper_nqvel]]).ravel()

    def _get_gripper_bias(self):
        return np.concatenate([self.sim.data.qfrc_bias[self.ur3_nqvel:self.ur3_nqvel+self.gripper_nqvel], 
            self.sim.data.qfrc_bias[2*self.ur3_nqvel+self.gripper_nqvel:2*self.ur3_nqvel+2*self.gripper_nqvel]]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

class URScriptWrapper(ActionWrapper):
    '''
    UR Script Wrapper for DualUR3Env
    '''
    def __init__(self, env, PID_gains, ur3_scale_factor, gripper_scale_factor):
        super().__init__(env)
        self.ur3_scale_factor = np.concatenate([ur3_scale_factor, ur3_scale_factor])
        self.gripper_scale_factor = np.concatenate([gripper_scale_factor, gripper_scale_factor])
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
        bias = self.env._get_gripper_bias() # Internal forces
        err = np.array([q[0], q[0], q[1], q[1]]) - self.env._get_gripper_qpos()
        action = scale_factor_gripper*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        return action
    
    def _velocityg(self, qd):
        assert q.shape[0] == self.ngripperdof
        bias = self.env._get_gripper_bias() # Internal forces
        err = np.array([qd[0], qd[0], qd[1], qd[1]]) - self.env._get_gripper_qvel()
        action = scale_factor_gripper*err + np.array([bias[2], bias[7], bias[12], bias[17]]) # P control
        return action

    def _forceg(self, qf):
        assert qf.shape[0] == self.ngripperdof
        bias = self.env._get_gripper_bias() # Internal forces
        action = np.array([qf[0], qf[0], qf[1], qf[1]]) + np.array([bias[2], bias[7], bias[12], bias[17]])
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)

    
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
