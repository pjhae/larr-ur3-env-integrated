import argparse
import numpy as np
import time
import sys

import gym_custom

from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3_LEFT as URScriptWrapper

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


def speedj_and_forceg(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('single-ur3-xy-left-larr-for-train-v0')
        speedj_args = {'a': 5, 't': None, 'wait': None}

    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_left = np.array([-0.45, -0.2, 0.8])  ## end-effector

    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.2, 'I': 10}} # was 0.2, 10.0
        ur3_scale_factor = np.array([24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    # Move to goal
    duration = 5.0 # in seconds
    obs_dict_current = env.env.get_obs_dict()
    
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/(duration*12)
    
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
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
    q_left_des_vel = np.zeros([env.ur3_nqpos])
    while qvel_err > np.deg2rad(1e0):
        obs, _, _, _ = env.step({
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)

        print('time: %f [s]'%(t*dt))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))

        qvel_err = np.linalg.norm(np.concatenate([obs_dict['left']['qvel']]) - np.concatenate([q_left_des_vel]))
        t += 1
    
    if env_type == list_of_env_types[0]:
        time.sleep(1)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()





if __name__ == '__main__':

    speedj_and_forceg(env_type='sim', render=True)

