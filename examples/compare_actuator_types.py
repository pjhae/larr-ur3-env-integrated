import argparse
import numpy as np
import time

import gym_custom

# print(gym_custom.envs.registry.all()) # List all registered environments
# print(env.model._joint_id2name) # Correspondence between qpos elements and joints


def motor_nogravity(desired_value, mode):
    env = gym_custom.make('Practice1-motor-nogravity-v0')
    obs = env.reset()
    dt = env.env.dt

    if mode == 'angle':
        # PID coefficients (Ziegler-Nichols method)
        K_u, T_u = 100.0, 2/3
        P_gain, I_gain, D_gain = 0.6*K_u, 1.2*K_u/T_u, 3*K_u*T_u/40
        err_integ = 0

        for _ in range(int(60/dt)):
            # Error
            theta_dist = np.mod(desired_value - obs[0], 2*np.pi)
            err = theta_dist - 2*np.pi*(theta_dist > np.pi)
            err_dot = -obs[1]
            err_integ = np.clip(err_integ + err*dt, -10, 10)

            # PID controller
            action = P_gain*err + I_gain*err_integ + D_gain*err_dot
            obs, _, _, _ = env.step(action)

            env.render()
            print('qpos: %f (degrees), qvel: %f (dps), qfrc_bias: %f'%(obs[0]*180/np.pi, obs[1]*180/np.pi, env.env.sim.data.qfrc_bias))
            time.sleep(dt)

    elif mode == 'velocity':
        # P coefficient
        K_u, T_u = 32.0, 0.04
        P_gain, I_gain = 0.45*K_u, 0.54*K_u/T_u
        err_integ = 0

        for _ in range(int(60/dt)):
            # Error
            err = desired_value - obs[1]
            err_integ = np.clip(err_integ + err*dt, -10, 10)

            # P controller
            action = P_gain*err + I_gain*err_integ
            obs, _, _, _ = env.step(action)

            env.render()
            print('qpos: %f (degrees), qvel: %f (dps), qfrc_bias: %f'%(obs[0]*180/np.pi, obs[1]*180/np.pi, env.env.sim.data.qfrc_bias))
            time.sleep(dt)

    else:
        raise ValueError('Invalid mode %s'%(mode))

    print('Times up!')

def motor_gravity_compensation(desired_value, mode):
    env = gym_custom.make('Practice1-motor-v0')
    obs = env.reset()
    dt = env.env.dt
    
    if mode == 'angle':
        # PID coefficients (Ziegler-Nichols method)
        K_u, T_u = 100.0, 2/3
        P_gain, I_gain, D_gain = 0.6*K_u, 1.2*K_u/T_u, 3*K_u*T_u/40
        # P_gain, I_gain, D_gain = 5.0, 0.0, 0.0
        err_integ = 0

        for _ in range(int(60/dt)):
            # Error
            theta_dist = np.mod(desired_value - obs[0], 2*np.pi)
            err = theta_dist - 2*np.pi*(theta_dist > np.pi)
            err_dot = -obs[1]
            err_integ = np.clip(err_integ + err*dt, -10, 10)

            # Internal forces 
            bias = env.env.sim.data.qfrc_bias*1.0

            # PID controller
            action = P_gain*err + I_gain*err_integ + D_gain*err_dot + bias
            obs, _, _, _ = env.step(action)

            env.render()
            print('qpos: %f (degrees), qvel: %f (dps), qfrc_bias: %f'%(obs[0]*180/np.pi, obs[1]*180/np.pi, env.env.sim.data.qfrc_bias))
            time.sleep(dt)

    elif mode == 'velocity':
        # P coefficient
        K_u, T_u = 32.0, 0.04
        P_gain, I_gain = 0.45*K_u, 0.54*K_u/T_u
        err_integ = 0

        for _ in range(int(60/dt)):
            # Error
            err = desired_value - obs[1]
            err_integ = np.clip(err_integ + err*dt, -10, 10)

            # Internal forces
            bias = env.env.sim.data.qfrc_bias*1.0

            # PI controller
            action = P_gain*err + I_gain*err_integ + bias
            obs, _, _, _ = env.step(action)

            env.render()
            print('qpos: %f (degrees), qvel: %f (dps), qfrc_bias: %f'%(obs[0]*180/np.pi, obs[1]*180/np.pi, env.env.sim.data.qfrc_bias))
            time.sleep(dt)

    else:
        raise ValueError('Invalid mode %s'%(mode))

    print('Times up!')

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
        print('qpos: %f (degrees), qvel: %f (dps)'%(obs[0]*180/np.pi, obs[1]*180/np.pi))
        time.sleep(dt)

    print('Times up!')

def position(desired_angle, dyntype=None):
    '''
    Position control (actuator/ position)
    '''
    if dyntype is None:
        env = gym_custom.make('Practice1-position-v0')
        obs = env.reset()
        dt = env.env.dt

        for _ in range(int(60/dt)):
            obs, _, _, _ = env.step(desired_angle)
            env.render()
            print('qpos: %f (degrees), qvel: %f (dps)'%(obs[0]*180/np.pi, obs[1]*180/np.pi))
            time.sleep(dt)
    elif dyntype == 'integrator':
        env = gym_custom.make('Practice1-position-dyntype-integrator-v0')
        obs = env.reset()
        dt = env.env.dt

        gain = 2.0
        for _ in range(int(60/dt)):
            theta_dist = np.mod(desired_angle - obs[0], 2*np.pi)
            err = theta_dist - 2*np.pi*(theta_dist > np.pi)
            # action = np.sign(err)
            action = np.clip(gain*err, -5, 5)

            obs, _, _, _ = env.step(action)
            env.render()
            print('qpos: %f (degrees), qvel: %f (dps)'%(obs[0]*180/np.pi, obs[1]*180/np.pi))
            time.sleep(dt)
    else:
        raise ValueError('Undefined dyntype')

    print('Times up!')

def velocity(desired_angular_velocity):
    '''
    Velocity control (actuator/ velocity)
    '''
    env = gym_custom.make('Practice1-velocity-v0')
    obs = env.reset()
    dt = env.env.dt

    for _ in range(int(60/dt)):
        obs, _, _, _ = env.step(desired_angular_velocity)
        env.render()
        print('qpos: %f (degrees), qvel: %f (dps)'%(obs[0]*180/np.pi, obs[1]*180/np.pi))
        time.sleep(dt)

    print('Times up!')

def no_dynamics(desired_angular_velocity):
    '''
    No dynamics (actuator/ none)
    '''
    env = gym_custom.make('Practice1-nodynamics-v0')
    obs = env.reset()
    dt = env.env.dt

    for _ in range(int(60/dt)):
        mjsimstate = env.env.sim.get_state()
        qpos, qvel = mjsimstate.qpos, mjsimstate.qvel
        qpos += desired_angular_velocity*dt
        qvel = desired_angular_velocity*np.ones_like(qvel)
        env.env.set_state(qpos, qvel)
        action = np.zeros_like(env.action_space.sample())
        # obs, _, _, _ = env.step(action)
        env.render()

        print('qpos: %f (degrees), qvel: %f (dps)'%(obs[0]*180/np.pi, obs[1]*180/np.pi))
        time.sleep(dt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare different types of MuJoCo actuators with Practice1-v0 envs.')
    actuator_type_list = ['motor', 'motor_nogravity_setangle', 'motor_nogravity_setvelocity', 
        'motor_gravity_compensation_setangle', 'motor_gravity_compensation_setvelocity',
        'position', 'position_dyntype_integrator', 'velocity', 'nodynamics'
    ]
    parser.add_argument('--actuator_type', type=str, required=False, default='motor_gravity_compensation_setvelocity',
        help='Specify an actuator type. Available types are: %s'%actuator_type_list
    )
    parser.add_argument('--desired_value', type=float, required=False, default=30,
        help='Desired angle(or angular velocity) of the pendulum in degrees(or degrees per second)'
    )
    args = parser.parse_args()

    if args.actuator_type == 'motor':
        motor(args.desired_value*np.pi/180.0)
    elif args.actuator_type == 'motor_nogravity_setangle':
        motor_nogravity(args.desired_value*np.pi/180.0, mode='angle')
    elif args.actuator_type == 'motor_nogravity_setvelocity':
        motor_nogravity(args.desired_value*np.pi/180.0, mode='velocity')
    elif args.actuator_type == 'motor_gravity_compensation_setangle':
        motor_gravity_compensation(args.desired_value*np.pi/180.0, mode='angle')
    elif args.actuator_type == 'motor_gravity_compensation_setvelocity':
        motor_gravity_compensation(args.desired_value*np.pi/180.0, mode='velocity')
    elif args.actuator_type == 'position':
        position(args.desired_value*np.pi/180.0)
    elif args.actuator_type == 'position_dyntype_integrator':
        position(args.desired_value*np.pi/180.0, dyntype='integrator')
    elif args.actuator_type == 'velocity':
        velocity(args.desired_value*np.pi/180.0)
    elif args.actuator_type == 'nodynamics':
        no_dynamics(args.desired_value*np.pi/180.0)
    else:
        raise ValueError('Specify an actuator type (--actuator_type). Available types are: %s'%actuator_type_list)