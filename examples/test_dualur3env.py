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

def test_fkine_ikine():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    # test forward kinematics
    q = env._get_ur3_qpos()[:env.ur3_nqpos] # right
    Rs, ps, Ts = env.forward_kinematics_DH(q, arm='right')
    R_base, p_base, T_base = env.get_body_se3(body_name='right_arm_rotz')
    # R_hand, p_hand, T_hand = env.get_body_se3(body_name='right_gripper:hand')
    R_hand, p_hand, T_hand = env.get_body_se3(body_name='right_gripper:hand')
    print('base:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[0,:], p_base))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[0,:,:], R_base))
    print('hand:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[-1,:], p_hand))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[-1,:,:], R_hand))

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    qpos_des = env.init_qpos.copy()
    qpos_des[0:env.ur3_nqpos] = q_right_des
    qpos_des[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des
    env.render()
    time.sleep(3.0)
    while True:
        env.set_state(qpos_des, env.init_qvel)
        env.render()

def servoj_and_forceg_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    PID_gains = {'P': 1.0, 'I': 0.5, 'D': 0.2}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    
    t = 0
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([1.0, 1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        t += 1
    time.sleep(100)

def servoj_and_forceg(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        servoj_args = {'t': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        servoj_args = {'t': 2/env.rate._freq, 'wait': False}
        # 1. Set initial as current configuration
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    if env_type == list_of_env_types[0]:
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    t = 0
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        t += 1
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        sys.exit()

def speedj_and_forceg_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    PI_gains = {'P': 0.20, 'I': 10.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    q_init = env.env._get_ur3_qpos()
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/3.0
    for t in range(int(3.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([1.0, 1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
    
    t = 0.0
    qvel = np.inf
    q_vel_des = np.zeros_like(env.env._get_ur3_qvel())
    while qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-1.0, -1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
        qvel = np.linalg.norm(env.env._get_ur3_qvel() - q_vel_des)
        t += 1
    time.sleep(100)

def speedj_and_forceg(env_type='sim', render=False):
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
        speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

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
    duration = 3.0 # in seconds
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

    # Stop
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
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()


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
        # 2. Set inital as default configuration
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

    if env_type == list_of_env_types[0]:
        curr_right_qpos = env.get_obs_dict()['right']['qpos']
        curr_left_qpos = env.get_obs_dict()['left']['qpos']

    null_obj_func = UprightConstraint()

    q_right_des = np.deg2rad([0, 0, 0, 0, 0, 45]) + curr_right_qpos
    q_left_des =  -np.deg2rad([0, 0, 0, 0, 0, 45]) + curr_left_qpos
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
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
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
        time.sleep(5)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()




def pick_and_place_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    PI_gains = {'P': 0.20, 'I': 5.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    # Move to position
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.0, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Open right gripper
    q_vel_des = np.zeros_like(q_vel_des)
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Place right gripper
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.0, -0.4, 0.78])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
    env.PID_gains = {'P': 1.0, 'I': 0.5, 'D': 0.2}
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        t += 1

    # Grip
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))

    # Lift right gripper
    env.PID_gains = {'P': 0.20, 'I': 5.0}
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.3, -0.5, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
    env.PID_gains = {'P': 1.0, 'I': 2.5, 'D': 0.2}
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        print('joint velocity [dps]: %f'%(qvel*180.0/np.pi))
        t += 1
    
    time.sleep(3.0)

    # Open gripper
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([-25.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('left arm joint error [deg]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
    
    while True:
        env.render()

def pick_and_place(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        # 1. Set initial as current configuration
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    if env_type == list_of_env_types[0]:
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':5.0}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    null_obj_func = UprightConstraint()

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Moving to position... (step 1 of 6)')
    time.sleep(1.0)
    # 1. Move to initial position
    duration = 5.0
    ee_pos_right = np.array([0.0, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    env.step({'right': {'stopj': {'a': speedj_args['a']}}, 'left': {'stopj': {'a': speedj_args['a']}}})
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Opening right gripper... (step 2 of 6)')
    time.sleep(1.0)
    env.step({'right': {'open_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    # 2. Open right gripper
    duration = 1.0
    q_right_des_vel, q_left_des_vel = np.zeros_like(q_right_des_vel), np.zeros_like(q_left_des_vel)
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Placing right gripper... (step 3 of 6)')
    time.sleep(1.0)
    # 3. Place right gripper
    duration = 5.0
    ee_pos_right = np.array([0.0, -0.4, 0.78])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(3e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        t += 1

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('grasp object?') is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Gripping object... (step 4 of 6)')
    time.sleep(1.0)
    env.step({'right': {'close_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    # 4. Grip object
    duration = 1.0
    start = time.time()
    for t in range(int(duration/dt)):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        # print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Lifting object... (step 5 of 6)')
    time.sleep(1.0)
    # 5. Lift object
    duration = 5.0
    duration = 5.0
    ee_pos_right = np.array([0.3, -0.5, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    if env_type == list_of_env_types[0]:
        env.wrapper_right.servoj_gains, env.wrapper_left.servoj_gains = {'P': 1.0, 'I': 2.5, 'D': 0.2}, {'P': 1.0, 'I': 2.5, 'D': 0.2}
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(3e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        print('joint velocity [dps]: %f'%(np.rad2deg(qvel)))
        t += 1

    time.sleep(3.0)

    print('Opening gripper... (step 6 of 6)')
    time.sleep(1.0)
    # 6. Open gripper
    duration = 1.0
    env.step({'right': {'open_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([-25.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
    print('done!')

def collide_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    PI_gains = {'P': 0.25, 'I': 10.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    # Move to position
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.15, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, -10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Collide with surface
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.15, -0.4, 0.69])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(60.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, -10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('    err_integ: %s'%(env.ur3_err_integ[:env.ur3_nqpos]))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
        print('    err_integ: %s'%(env.ur3_err_integ[-env.ur3_nqpos:]))

def collide(env_type='sim', render=False):     # You should activate this function only on simulation
    list_of_env_types = ['sim']

    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        speedj_args = {'a': 5, 't': None, 'wait': None}
    # elif env_type == list_of_env_types[1]:
    #     env = gym_custom.make('dual-ur3-larr-real-v0')
    #     speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.25, 'I': 10.0}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    # Move to goal
    duration = 5.0 # in seconds
    ee_pos_right = np.array([0.15, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
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

    # Collide with surface
    ee_pos_right = np.array([0.15, -0.4, 0.69])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    for t in range(int(10*duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
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
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('    err_integ: %s'%(env.wrapper_right.ur3_err_integ))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
            print('    err_integ: %s'%(env.wrapper_left.ur3_err_integ))
    
def real_env_get_obs_rate_test(wait=True):
    env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
    stime = time.time()
    [env._get_obs(wait=wait) for _ in range(100)]
    ftime = time.time()
    # stats
    # single call: 8ms (wait=True, default), <1ms (wait=False)
    # 2 calls: 16ms (wait=True, default), <1ms (wait=False)
    # 3 calls: 17ms (wait=True, default), <1ms (wait=False)
    # 4 calls: 24ms (wait=True, default), <1ms (wait=False)
    print('\r\ntime per call: %f ms'%((ftime-stime)/100*1000))
    print('done!\r\n')

def real_env_command_send_rate_test(wait=True):
    # ~25ms (wait=True, default), ~11ms (wait=False)
    env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=100
        )
    env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
    env.set_initial_gripper_pos(np.array([0.0, 0.0]))
    env.reset()
    command = {
        'right': {'speedj': {'qd': np.zeros([6]), 'a': 1.0, 't': 1.0, 'wait': False}},
        'left': {'speedj': {'qd': np.zeros([6]), 'a': 1.0, 't': 1.0, 'wait': False}}
    }
    stime = time.time()
    [env.step(command, wait=wait) for _ in range(100)]
    ftime = time.time()
    command = {
        'right': {'stopj': {'a': 1.0}},
        'left': {'stopj': {'a': 1.0}}
    }
    env.step(command)
    print('\r\ntime per call: %f ms'%((ftime-stime)/100*1000))
    print('done!\r\n')

if __name__ == '__main__':
    # 1. MuJoCo model verification
    # show_dual_ur3()
    # run_dual_ur3()
    # test_fkine_ikine()

    # 2.1 Updated UR wrapper examples
    # servoj_and_forceg(env_type='real', render=False)
    # speedj_and_forceg(env_type='real', render=False)
    # pick_and_place(env_type='sim', render=True)
    # collide(env_type='sim', render=True)
    fidget_in_place(env_type='sim', render=True)

    # 2.2 Deprecated UR wrapper examples 
    # servoj_and_forceg_deprecated()
    # speedj_and_forceg_deprecated()
    # pick_and_place_deprecated()
    # collide_deprecated()

    # 3. Misc. tests
    # real_env_get_obs_rate_test(wait=False)
    # real_env_command_send_rate_test(wait=False)
    # pass