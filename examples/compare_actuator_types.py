import argparse
import numpy as np
import time

import gym_custom

# print(gym_custom.envs.registry.all()) # List all registered environments
# print(env.model._joint_id2name) # Correspondence between qpos elements and joints


def motor(desired_angle):
    '''
    Torque control (actuator/ motor)
    '''
    env = gym_custom.make('Practice1-motor-v0')
    obs = env.reset()
    dt = env.env.dt

    # PID coefficients (Ziegler–Nichols method)
    K_u, T_u = 100.0, 2/3
    P_gain, I_gain, D_gain = 0.6*K_u, 1.2*K_u/T_u, 3*K_u*T_u/40
    err_integ = 0

    for _ in range(int(60/dt)):
        # Error
        theta_dist = np.mod(desired_angle - obs[0], 2*np.pi)
        err = theta_dist - 2*np.pi*(theta_dist > np.pi)
        err_dot = -obs[1]
        err_integ = np.clip(err_integ + err*dt, -10, 10)

        # PID controller
        action = P_gain*err + I_gain*err_integ + D_gain*err_dot
        obs, _, _, _ = env.step(action)

        env.render()
        time.sleep(dt)

    print('Times up!')

def position(desired_angle, dyntype=None):
    if dyntype is None:
        env = gym_custom.make('Practice1-position-v0')
    elif dyntype == 'integrator':
        env = gym_custom.make('Practice1-position-dyntype-integrator-v0')
    else:
        raise ValueError('Undefined dyntype')

    print('Times up!')

def velocity(desired_angular_velocity):
    env = gym_custom.make('Practice1-velocity-v0')

    print('Times up!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare different types of MuJoCo actuators with Practice1-v0 envs.')
    actuator_type_list = ['motor', 'position', 'position(dyntype=integrator)', 'velocity']
    parser.add_argument('--actuator_type', type=str, default='velocity',
        help='Specify an actuator type. Available types are: %s'%actuator_type_list
    )
    parser.add_argument('--desired_value', type=float, default=0.0,
        help='Desired angle(or angular velocity) of the pendulum in degrees(or degrees per second)'
    )
    args = parser.parse_args()

    if args.actuator_type == 'motor':
        motor(args.desired_value*np.pi/180.0)
    elif args.actuator_type == 'position':
        position(args.desired_value*np.pi/180.0)
    elif args.actuator_type == 'position(dyntype=integrator)':
        position(args.desired_value*np.pi/180.0, dyntype='integrator')
    elif args.actuator_type == 'velocity':
        velocity(args.desired_value*np.pi/180.0)
    else:
        raise ValueError('Specify an actuator type (--actuator_type). Available types are: %s'%actuator_type_list)