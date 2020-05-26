import argparse
import numpy as np
import time

import gym_custom

class PositionControlWrapper(gym_custom.core.ActionWrapper):
    '''
    Gravity-compensated joint position PID controller wrapper
    '''
    def __init__(self, env, control_gains, scale_factor, ndof):
        super().__init__(env)
        self.P_gain = control_gains['P']
        self.I_gain = control_gains['I']
        self.D_gain = control_gains['D']
        self.scale_factor = scale_factor
        self.ndof = ndof
        self.err_integ = 0.0
    
    def action(self, desired_theta, relative=False):
        # Error
        current_theta = self.env.sim.data.qpos[:self.ndof]
        if relative: # Relative error
            theta_dist = np.mod(desired_theta - current_theta, 2*np.pi)
            err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        else: # Absolute error
            err = desired_theta - current_theta
        err_dot = -self.env.sim.data.qvel[:self.ndof]
        self.err_integ = np.clip(self.err_integ + err*self.env.dt, -100, 100)

        # Internal forces 
        bias = self.env.sim.data.qfrc_bias[:self.ndof]*1.0

        # PID controller
        action = self.scale_factor*(self.P_gain*err + self.I_gain*self.err_integ + self.D_gain*err_dot) + bias
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)

class PositionControlWrapperwithGripper(PositionControlWrapper):
    
    def __init__(self, env, control_gains, scale_factor, ndof, ngripperdof):
        assert scale_factor.shape[0] == ndof + ngripperdof
        super().__init__(env, control_gains, scale_factor[:ndof], ndof)
        self.scale_factor_gripper = scale_factor[-ngripperdof:]
        self.ngripperdof = ngripperdof

    def action(self, desired_values):
        joint_action = super().action(desired_values['theta'])
        gripper_action = desired_values['gripperpos']*np.ones([self.ngripperdof])
        return np.concatenate([joint_action, gripper_action])

class VelocityControlWrapper(gym_custom.core.ActionWrapper):
    '''
    Gravity-compensated joint velocity PID controller wrapper
    '''
    def __init__(self, env, control_gains, scale_factor, ndof):
        super().__init__(env)
        self.P_gain = control_gains['P']
        self.I_gain = control_gains['I']
        self.scale_factor = scale_factor
        self.ndof = ndof
        self.err_integ = 0.0

    def action(self, desired_thetadot):
        # Error
        current_thetadot = self.env.sim.data.qvel[:self.ndof]
        err = desired_thetadot - current_thetadot
        self.err_integ = np.clip(self.err_integ + err*self.env.dt, -100, 100)

        # Internal forces
        bias = self.env.sim.data.qfrc_bias[:self.ndof]*1.0

        # PID controller
        action = self.scale_factor*(self.P_gain*err + self.I_gain*self.err_integ) + bias
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)

class VelocityControlWrapperwithGripper(VelocityControlWrapper):
    
    def __init__(self, env, control_gains, scale_factor, ndof, ngripperdof):
        assert scale_factor.shape[0] == ndof + ngripperdof
        super().__init__(env, control_gains, scale_factor[:ndof], ndof)
        self.scale_factor_gripper = scale_factor[-ngripperdof:]
        self.ngripperdof = ngripperdof

    def action(self, desired_values):
        joint_action = super().action(desired_values['thetadot'])
        gripper_action = desired_values['gripperpos']*np.ones([self.ngripperdof])
        return np.concatenate([joint_action, gripper_action])

def random_action():
    env = gym_custom.make('ur3-practice-v0')
    obs = env.reset()
    dt = env.dt

    for _ in range(int(60/dt)):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        qpos, qvel = obs[:env.model.nq], obs[-env.model.nv:]

        env.render()
        print('qpos: %s (degrees), qvel: %s (dps), qfrc_bias: %s'%(qpos[:6]*180/np.pi, qvel[:6]*180/np.pi, env.sim.data.qfrc_bias[:6]))
        time.sleep(dt)
    
    print('Times up!')

def set_state(state='home'): # kinematics only
    env = gym_custom.make('ur3-practice-v0')
    obs = env.reset()
    dt = env.dt

    for t in range(int(60/dt)):
        mjsimstate = env.sim.get_state()
        qpos, qvel = mjsimstate.qpos, mjsimstate.qvel
        if state == 'home':
            pos = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]) # Home position
        elif state == 'zero':
            pos = np.zeros([6]) # Zero position
        else:
            raise ValueError('Undefined state')
        qpos[:6] = pos
        qvel[:6] = np.zeros([6])
        env.set_state(qpos, qvel)

        env.render()
        print('time: %.2f'%(t*dt))
        print('  qpos: %s (degrees)'%(qpos[:6]*180/np.pi))
        print('  qvel: %s (dps)'%(qvel[:6]*180/np.pi))
        print('  qfrc_bias: %s'%(env.sim.data.qfrc_bias[:6]))
        time.sleep(dt)

    print('Times up!')

def set_theta(): # movej
    env = gym_custom.make('ur3-practice-v0')
    obs = env.reset()
    dt = env.dt

    PID_gains = {'P': 5.0, 'I': 0.0, 'D': 0.0}
    scale = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])
    env = PositionControlWrapper(env, control_gains=PID_gains, scale_factor=scale, ndof=6)
    
    # desired_theta = np.zeros([6])
    # desired_theta = np.array([-60.0, -90.0, 30.0, -90.0, 90.0, 0.0])*np.pi/180
    desired_theta = np.array([270.0, -90.0, -90.0, -90.0, 90.0, 180.0])*np.pi/180.0
    for t in range(int(60/dt)):
        obs, _, _, _ = env.step(desired_theta)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        theta, theta_dot = qpos[:6], qvel[:6]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  qpos: %s (degrees)'%(theta*180/np.pi))
        print('  qvel: %s (dps)'%(theta_dot*180/np.pi))
        print('  qfrc_bias: %s'%(env.sim.data.qfrc_bias[:6]))
        time.sleep(dt)

    print('Times up!')
    time.sleep(120)

def set_thetadot(): # speedj
    env = gym_custom.make('ur3-practice-v0')
    obs = env.reset()
    dt = env.dt

    PI_gains = {'P': 0.20, 'I': 5.0}
    scale = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    env = VelocityControlWrapper(env, control_gains=PI_gains, scale_factor=scale, ndof=6)

    # desired_theta_dot = np.array([18.0, 0.0, 0.0, 0.0, 0.0, 0.0])*np.pi/180
    desired_theta_dot = np.array([27.0, -9.0, -9.0, -9.0, 9.0, 18.0])*np.pi/180 # stress test
    for t in range(int(10/dt)):
        obs, _, _, _ = env.step(desired_theta_dot)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        theta, theta_dot = qpos[:6], qvel[:6]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  qpos: %s (degrees)'%(theta*180/np.pi))
        print('  qvel: %s (dps)'%(theta_dot*180/np.pi))
        print('  qfrc_bias: %s'%(env.sim.data.qfrc_bias[:6]))
        time.sleep(dt)

    print('Times up!')
    time.sleep(120)

def control_gripper():
    env = gym_custom.make('ur3-practice-withgripper-v0')
    obs = env.reset()
    dt = env.dt
    
    PI_gains = {'P': 0.20, 'I': 5.0}
    scale = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0, 1.0, 1.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5, 1.0, 1.0])
    env = VelocityControlWrapperwithGripper(env, control_gains=PI_gains, scale_factor=scale, ndof=6, ngripperdof=2)

    # Move arm
    desired_theta_dot = np.array([0.0, -9.0, 0.0, 0.0, 0.0, 0.0])*np.pi/180
    for t in range(int(10/dt)):
        desired_values = {'thetadot': desired_theta_dot, 'gripperpos': 0.0}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        theta, theta_dot = qpos[:6], qvel[:6]
        gripper, gripper_dot = qpos[6:14], qvel[6:14]
        mug, mug_dot = qpos[14:21], qvel[14:20]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  qpos: %s (degrees)'%(theta*180/np.pi))
        print('  qvel: %s (dps)'%(theta_dot*180/np.pi))
        print('  qfrc_bias: %s (?)'%(env.sim.data.qfrc_bias[:6]))
        print('  gripper: %s (degrees)'%(gripper[[0,4]]*180/np.pi))
        print('  grippervel: %s (dps)'%(gripper_dot[[0,4]]*180/np.pi))
        print('  gripperbias: %s (?)'%(env.sim.data.qfrc_bias[[6,10]]))
        # time.sleep(dt)

    print('Opening gripper in 5 secs..')
    time.sleep(5)
    # Open gripper
    for t in range(int(5/dt)):
        desired_values = {'thetadot': np.zeros([6]), 'gripperpos': -1.0}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        gripper, gripper_dot = qpos[6:14], qvel[6:14]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  gripper: %s (degrees)'%(gripper[[0,4]]*180/np.pi))
        print('  grippervel: %s (dps)'%(gripper_dot[[0,4]]*180/np.pi))
        print('  gripperbias: %s (?)'%(env.sim.data.qfrc_bias[[6,10]]))
        # time.sleep(dt)

    print('Closing gripper in 5 secs..')
    time.sleep(5)
    # Close gripper
    for t in range(int(5/dt)):
        desired_values = {'thetadot': np.zeros([6]), 'gripperpos': 1.0}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        gripper, gripper_dot = qpos[6:14], qvel[6:14]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  gripper: %s (degrees)'%(gripper[[0,4]]*180/np.pi))
        print('  grippervel: %s (dps)'%(gripper_dot[[0,4]]*180/np.pi))
        print('  gripperbias: %s (?)'%(env.sim.data.qfrc_bias[[6,10]]))
        # time.sleep(dt)

    time.sleep(120)

def gripper_grasp_test():
    env = gym_custom.make('robotiq-practice-v0')
    obs = env.reset()
    dt = env.dt

    PID_gains = {'P': 10, 'I': 5, 'D': 2}
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    env = PositionControlWrapperwithGripper(env, control_gains=PID_gains, scale_factor=scale, ndof=3, ngripperdof=2)

    num_pole_joints, num_pole_jointvels = 3, 3
    num_gripper_joints, num_gripper_jointvels = 2, 2
    num_object_joints, num_object_jointvels = 7, 6
    num_qpos = [num_pole_joints, num_gripper_joints, num_object_joints]
    num_qvel = [num_pole_jointvels, num_gripper_jointvels, num_object_jointvels]

    # Open gripper
    for t in range(int(5/dt)):
        pole_xyz = np.array([0.0, 0.0, -0.04])
        # gripper_pos = -1
        gripper_pos = -2
        gripper_pos = 0
        desired_values = {'theta': pole_xyz, 'gripperpos': gripper_pos}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        pole, pole_dot = qpos[:sum(num_qpos[:1])], qvel[:sum(num_qvel[:1])]
        gripper, gripper_dot = qpos[sum(num_qpos[:1]):sum(num_qpos[:2])], qvel[sum(num_qvel[:1]):sum(num_qvel[:2])]
        mug, mug_dot = qpos[sum(num_qpos[:2]):sum(num_qpos[:3])], qvel[sum(num_qvel[:2]):sum(num_qvel[:3])]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  pole_pos: %s (m)'%(pole))
        print('  pole_vel: %s (m/s)'%(pole_dot))
        print('  pole_bias: %s (N)'%(env.sim.data.qfrc_bias[:sum(num_qpos[:1])]))
        print('  gripper_pos: %s (degrees)'%(gripper[[0,num_gripper_joints-1]]*180/np.pi))
        print('  gripper_vel: %s (dps)'%(gripper_dot[[0,num_gripper_jointvels-1]]*180/np.pi))
        print('  gripper_bias: %s (Nm)'%(env.sim.data.qfrc_bias[[sum(num_qvel[:1]),sum(num_qvel[:2])]]))
        time.sleep(dt)

    # Close gripper
    for t in range(int(5/dt)):
        pole_xyz = np.array([0.0, 0.0, -0.04])
        # gripper_pos = 1
        gripper_pos = 20
        desired_values = {'theta': pole_xyz, 'gripperpos': gripper_pos}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        pole, pole_dot = qpos[:sum(num_qpos[:1])], qvel[:sum(num_qvel[:1])]
        gripper, gripper_dot = qpos[sum(num_qpos[:1]):sum(num_qpos[:2])], qvel[sum(num_qvel[:1]):sum(num_qvel[:2])]
        mug, mug_dot = qpos[sum(num_qpos[:2]):sum(num_qpos[:3])], qvel[sum(num_qvel[:2]):sum(num_qvel[:3])]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  pole_pos: %s (m)'%(pole))
        print('  pole_vel: %s (m/s)'%(pole_dot))
        print('  pole_bias: %s (N)'%(env.sim.data.qfrc_bias[:sum(num_qpos[:1])]))
        print('  gripper_pos: %s (degrees)'%(gripper[[0,num_gripper_joints-1]]*180/np.pi))
        print('  gripper_vel: %s (dps)'%(gripper_dot[[0,num_gripper_jointvels-1]]*180/np.pi))
        print('  gripper_bias: %s (Nm)'%(env.sim.data.qfrc_bias[[sum(num_qvel[:1]),sum(num_qvel[:2])]]))
        time.sleep(dt)

    # Lift gripper
    for t in range(int(10/dt)):
        pole_xyz = np.array([0.0, 0.0, 0.20])
        # gripper_pos = 1
        gripper_pos = 50
        desired_values = {'theta': pole_xyz, 'gripperpos': gripper_pos}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        pole, pole_dot = qpos[:sum(num_qpos[:1])], qvel[:sum(num_qvel[:1])]
        gripper, gripper_dot = qpos[sum(num_qpos[:1]):sum(num_qpos[:2])], qvel[sum(num_qvel[:1]):sum(num_qvel[:2])]
        mug, mug_dot = qpos[sum(num_qpos[:2]):sum(num_qpos[:3])], qvel[sum(num_qvel[:2]):sum(num_qvel[:3])]

        env.render()
        print('time: %.2f'%(t*dt))
        print('  pole_pos: %s (m)'%(pole))
        print('  pole_vel: %s (m/s)'%(pole_dot))
        print('  pole_bias: %s (N)'%(env.sim.data.qfrc_bias[:sum(num_qpos[:1])]))
        print('  gripper_pos: %s (degrees)'%(gripper[[0,num_gripper_joints-1]]*180/np.pi))
        print('  gripper_vel: %s (dps)'%(gripper_dot[[0,num_gripper_jointvels-1]]*180/np.pi))
        print('  gripper_bias: %s (Nm)'%(env.sim.data.qfrc_bias[[sum(num_qvel[:1]),sum(num_qvel[:2])]]))
        time.sleep(dt)

    # Release gripper
    for t in range(int(5/dt)):
        pole_xyz = np.array([0.0, 0.0, 0.20])
        gripper_pos = -1
        desired_values = {'theta': pole_xyz, 'gripperpos': gripper_pos}
        obs, _, _, _ = env.step(desired_values)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        pole, pole_dot = qpos[:sum(num_qpos[:1])], qvel[:sum(num_qvel[:1])]
        gripper, gripper_dot = qpos[sum(num_qpos[:1]):sum(num_qpos[:2])], qvel[sum(num_qvel[:1]):sum(num_qvel[:2])]
        mug, mug_dot = qpos[sum(num_qpos[:2]):sum(num_qpos[:3])], qvel[sum(num_qvel[:2]):sum(num_qvel[:3])]
        
        env.render()
        print('time: %.2f'%(t*dt))
        print('  pole_pos: %s (m)'%(pole))
        print('  pole_vel: %s (m/s)'%(pole_dot))
        print('  pole_bias: %s (N)'%(env.sim.data.qfrc_bias[:sum(num_qpos[:1])]))
        print('  gripper_pos: %s (degrees)'%(gripper[[0,num_gripper_joints-1]]*180/np.pi))
        print('  gripper_vel: %s (dps)'%(gripper_dot[[0,num_gripper_jointvels-1]]*180/np.pi))
        print('  gripper_bias: %s (Nm)'%(env.sim.data.qfrc_bias[[sum(num_qvel[:1]),sum(num_qvel[:2])]]))
        time.sleep(dt)

    time.sleep(120)

def grasp_mug():
    pass

if __name__ == '__main__':
    # random_action()
    # set_home()
    # set_theta()
    # set_thetadot()
    # control_gripper()
    gripper_grasp_test()
    # grasp_mug()
