import argparse
import numpy as np
import time

import gym_custom

class PositionControlWrapper(gym_custom.core.ActionWrapper):
    '''
    Gravity-compensated joint position PID controller wrapper
    '''
    def __init__(self, env, control_gains):
        super().__init__(env)
        self.P_gain = control_gains['P']
        self.I_gain = control_gains['I']
        self.D_gain = control_gains['D']
        self.err_integ = 0.0
    
    def action(self, desired_position):
        # Error
        current_position = self.env.sim.data.qpos
        theta_dist = np.mod(desired_position - current_position, 2*np.pi)
        err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        err_dot = -self.env.sim.data.qvel
        self.err_integ = np.clip(self.err_integ + err*self.env.dt, -100, 100)

        # Internal forces 
        bias = env.env.sim.data.qfrc_bias

        # PID controller
        action = self.P_gain*err + self.I_gain*self.err_integ + self.D_gain*err_dot + bias
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)

class VelocityControlWrapper(gym_custom.core.ActionWrapper):
    '''
    Gravity-compensated joint velocity PID controller wrapper
    '''
    def __init__(self, env, control_gains):
        super().__init__(env)
        self.P_gain = control_gains['P']
        self.I_gain = control_gains['I']
        self.err_integ = 0.0

    def action(self, desired_velocity):
        # Error
        current_velocity = self.env.sim.data.qvel
        err = desired_velocity - current_velocity
        self.err_integ = np.clip(self.err_integ + err*self.env.dt, -100, 100)

        # Internal forces
        bias = env.env.sim.data.qfrc_bias

        # PID controller
        action = self.P_gain*err + self.I_gain*self.err_integ + bias
        return action

    def reset(self, **kwargs):
        self.err_integ = 0.0
        return self.env.reset(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ActionWrapper for position and velocity.')
    env_list = ['Practice1-motor-notimelimit-v0', 'Practice2-motor-notimelimit-v0']
    control_type_list = ['position', 'velocity']
    parser.add_argument('--env_name', type=str, required=False, default=env_list[1],
        help='Specify an environment. Available options are: %s'%env_list
    )
    parser.add_argument('--control_type', type=str, required=False, default=control_type_list[1],
        help='Specify control type. Available options are: %s'%control_type_list
    )
    parser.add_argument('--desired_value', type=str, required=False, default=360.0,
        help='Desired angle(or angular velocity) of the pendulum in degrees(or degrees per second)'
    )
    args = parser.parse_args()
    assert args.env_name in env_list, 'Invalid option for --env_name! Available options are: %s'%env_list
    assert args.control_type in control_type_list, 'Invalid option for --control_type! Available options are: %s'%control_type_list

    env = gym_custom.make(args.env_name)
    obs = env.reset()
    dt = env.dt
    
    if args.control_type == 'position':
        if args.env_name == env_list[0]:
            # PID coefficients (Ziegler-Nichols method)
            K_u, T_u = 100.0, 2/3
            PID_gains = {'P': 0.6*K_u, 'I': 1.2*K_u/T_u, 'D': 3*K_u*T_u/40}
        elif args.env_name == env_list[1]:
            # PID coefficients
            PID_gains = {'P': 25, 'I': 10, 'D': 5}
        env = PositionControlWrapper(env, PID_gains)
    elif args.control_type == 'velocity':
        if args.env_name == env_list[0]:
            # PI coefficients (Ziegler-Nichols method)
            K_u, T_u = 32.0, 0.04
            PI_gains = {'P': 0.45*K_u, 'I': 0.54*K_u/T_u}
        elif args.env_name == env_list[1]:
            PI_gains = {'P': 20, 'I': 0}
        env = VelocityControlWrapper(env, PI_gains)

    for _ in range(int(60/dt)):
        obs, _, _, _ = env.step(args.desired_value*np.pi/180.0)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]

        env.render()
        print('qpos: %s (degrees), qvel: %s (dps), qfrc_bias: %s'%(qpos*180/np.pi, qvel*180/np.pi, env.env.sim.data.qfrc_bias))
        time.sleep(dt)