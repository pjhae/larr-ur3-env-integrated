import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import time
import sys
import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3 as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no
from collections import OrderedDict
from utils import save_data, load_data
import os
import os.path as osp

# choose the env
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--exp_type', default="sim", help='choose sim or real')
args = parser.parse_args()

# Simulation Environment
env = gym_custom.make('single-ur3-larr-for-train-v0')
servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
gripper_scale_factor = np.array([1.0])
env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)


# Real Environment
if args.exp_type == 'real':
    real_env = gym_custom.make('single-ur3-larr-real-for-train-v0',
        host_ip_right='192.168.5.102',
        rate=25
    )
    servoj_args, speedj_args = {'t': 2/real_env.rate._freq, 'wait': False}, {'a': 0.01, 't': 2/real_env.rate._freq, 'wait': False}
    # 1. Set initial as current configuration
    real_env.set_initial_joint_pos('current')
    real_env.set_initial_gripper_pos('current')
    # 2. Set inital as default configuration
    real_env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0]))
    real_env.set_initial_gripper_pos(np.array([0.0]))
    obs = env.reset()
    dt = env.dt
    time.sleep(1.0)


# Action limits 참고용
COMMAND_LIMITS = {
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -1])*0.25,
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1])*0.25], # [rad/s]
}

# Pre-defined action sequence
action_seq = np.array([[-0.5,-0.5,-1,-1,-1,-1.2,-1,-1]]*500+[[0.5,0.5,1,1,1,1.2,1,1]]*500+\
                      [[-0.5,-0.5,-1,-1,-1,-1.2,-1,-1]]*500+[[0.5,0.5,1,1,1,1.2,1,1]]*500+[[-0.5,-0.5,-1,-1,-1,-1.2,-1,-1]]*500)


# Run simulation

# if real, get the data
if args.exp_type == 'real':
    real_data = []
    state = env.reset()
    for i in range(2500):
        next_state, reward, done, _  = env.step({
            'right': {
                'speedj': {'qd': action_seq[i][:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([action_seq[i][6]])}}
        })
        curr_pos = real_env.get_obs_dict()['right']['curr_pos']      # from real env
        real_data.append(curr_pos)
    # Save real data
    real_data = np.array(real_data)
    save_data(real_data, "real_data.npy")


# if sim, RUN CEM
else:
    n_seq = 10
    n_horrizon = 2500
    n_dim = 3
    n_iter = 10
    n_elit = 3

    # a, P, I params
    lim_high = np.array([10, 1, 20])
    lim_low  = np.array([0, 0, 0])

    # load data
    sim_data = np.zeros([n_seq, n_horrizon, n_dim])
    real_data = load_data("sac_practice/data/real_data.npy")

    # CEM
    for k in range(n_iter):

        # sample params
        if k == 0:
            candidate_parameters = lim_low + (lim_high - lim_low)*np.random.rand(n_seq, len(lim_high))
            prams_mean = np.mean(candidate_parameters, axis=0)
            prams_std = np.std(candidate_parameters, axis=0)
        else:
            candidate_parameters = np.random.normal(prams_mean, prams_std, (n_seq, len(lim_high)))

        # evaluate params
        for i in range(n_seq):
            state = env.reset()
            env.wrapper_right.speedj_gains['P'] = candidate_parameters[i][1]
            env.wrapper_right.speedj_gains['I'] = candidate_parameters[i][2]
            for j in range(n_horrizon):
                curr_pos = env.get_obs_dict()['right']['curr_pos']       # from sim env
                sim_data[i][j][:] = curr_pos
                next_state, reward, done, _  = env.step({
                'right': {
                    'speedj': {'qd':  action_seq[j][:6], 'a': candidate_parameters[i][0], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([action_seq[j][6]])}
                    }
                })
                env.render()


        mse_results = np.zeros(n_seq)
        for i in range(n_seq):
            mse = np.mean((sim_data[i] - real_data) ** 2)
            mse_results[i] = mse

        print("MSE 결과 배열:", mse_results)
        smallest_indices = np.argpartition(mse_results, n_elit)[:n_elit]

        print("가장 작은 MSE 값들의 인덱스:", smallest_indices)













