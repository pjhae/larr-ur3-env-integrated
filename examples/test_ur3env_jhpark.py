
import argparse
import numpy as np
import time
import sys

import gym_custom

from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3_deprecated as URScriptWrapper_deprecated
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3 as URScriptWrapper
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no


class NoConstraint(NullObjectiveBase):
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        return 0.0


class UprightConstraint(NullObjectiveBase):
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)

def show_dual_ur3():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    while True:
        env.render()
        time.sleep(dt)

def run_dual_ur3():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    for t in range(int(60/dt)):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        env.render()
        print('time: %.2f'%(t*dt))
        print('  joint_pos: %s (rad)'%(env._get_ur3_qpos()*180/np.pi))
        print('  joint_vel: %s (rad/s)'%(env._get_ur3_qvel()*180/np.pi))
        print('  joint_bias: %s (Nm)'%(env._get_ur3_bias()*180/np.pi))
        print('  gripper_pos: %s (m)'%(env._get_gripper_qpos()[[2,7,12,17]]))
        print('  gripper_vel: %s (m/s)'%(env._get_gripper_qvel()[[2,7,12,17]]))
        print('  gripper_bias: %s (N)'%(env._get_gripper_bias()[[2,7,12,17]]))
        time.sleep(dt)


def fidget_in_place(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        speedj_args = {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        env.interface_right.reset_controller()
        env.interface_left.reset_controller()
        right_status = env.interface_right.get_controller_status()
        left_status = env.interface_left.get_controller_status()
        controller_error = lambda stats: np.any([(stat.safety.StoppedDueToSafety) or (not stat.robot.PowerOn) for stat in stats])
        if prompt_yes_or_no('controller status ok? \r\n right: %s\r\n left: %s\r\n?'
            %(not controller_error([right_status]), not controller_error([left_status]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

        speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')

        # Set inital as default configuration
        curr_right_qpos, curr_left_qpos = env.interface_right.get_joint_positions(), env.interface_left.get_joint_positions()
        curr_right_gripper_pos, curr_left_gripper_pos = env.interface_right.get_gripper_position(), env.interface_left.get_gripper_position()
        print('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n'
            %(np.rad2deg(curr_right_qpos), np.rad2deg(curr_left_qpos)))
        print('current gripper_pos is \r\n right: %s deg\r\n left: %s deg\r\n'
            %(np.rad2deg(curr_right_gripper_pos), np.rad2deg(curr_left_gripper_pos)))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    # q_right_des = curr_right_qpos + np.deg2rad([5, 5, 10, 10, 10, 90])       # JONGHAE
    # q_left_des = curr_left_qpos + np.deg2rad([5, 5, 10, 10, 10, 90])
    # gripper_right_des = 1 - curr_right_gripper_pos
    # gripper_left_des = 1 - curr_left_gripper_pos

    q_right_des =  + env.get_obs_dict()['right']['qpos']
    q_left_des =  + env.get_obs_dict()['left']['qpos']
    gripper_right_des = 1 
    gripper_left_des = 1 

    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 10.0
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    # Move to goal
    duration = 5.0 # in seconds
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            }
        })
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    # Stop (see q_vel is np.zeros([]) ) 
    t = 0
    qvel_err = np.inf
    q_right_des_vel, q_left_des_vel = np.zeros([env.ur3_nqpos]), np.zeros([env.ur3_nqpos])
    while qvel_err > np.deg2rad(1e0):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        qvel_err = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]) - np.concatenate([q_right_des_vel, q_left_des_vel]))
        t += 1

    # Check gripper (close/open)
    # time.sleep(1.0)
    # env.step({'right': {'close_gripper': {}}, 'left': {'close_gripper': {}}})
    # time.sleep(3.0)
    # env.step({'right': {'open_gripper': {}}, 'left': {'open_gripper': {}}})
    # time.sleep(3.0)
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()


if __name__ == '__main__':

    # simple task
    fidget_in_place(env_type='sim', render=True)

