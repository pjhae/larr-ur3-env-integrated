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
import os
import os.path as osp



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
# choose the env
parser.add_argument('--env_type', default="sim",
                    help='choose sim or real')
args = parser.parse_args()


# Episode to test
num_epi = 240

# Rendering (if env_type is real, render should be FALSE)
render = True

# Environment
if args.env_type == "sim":
    env = gym_custom.make('single-ur3-larr-for-train-v0')
    servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}

elif args.env_type == "real":
    env = gym_custom.make('single-ur3-larr-real-for-train-v0',
        host_ip_right='192.168.5.102',
        rate=25
    )
    servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 0.01, 't': 2/env.rate._freq, 'wait': False}
    # 1. Set initial as current configuration
    env.set_initial_joint_pos('current')
    env.set_initial_gripper_pos('current')
    # 2. Set inital as default configuration
    env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0]))
    env.set_initial_gripper_pos(np.array([0.0]))
    assert render is False

else:
    print("Please choose sim or real")

obs = env.reset()
dt = env.dt

if args.env_type == "sim":
    PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
elif args.env_type == "real":
        env.env = env

if args.env_type == "real":
    if prompt_yes_or_no('current qpos is \r\n right: %s deg?\r\n'
        %(np.rad2deg(env.env._init_qpos[:6]))) is False:
        print('exiting program!')
        env.close()
        sys.exit()
time.sleep(1.0)

# Seed
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


COMMAND_LIMITS = {
    'movej': [np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -np.inf]),
        np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, np.inf])], # [rad]
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -1])*0.20,
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1])*0.20], # [rad/s]
    'move_gripper': [np.array([-1]), np.array([1])] # [0: open, 1: close]
}

def convert_action_to_space(action_limits):
    if isinstance(action_limits, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_action_to_space(value))
            for key, value in COMMAND_LIMITS.items()
        ]))
    elif isinstance(action_limits, list):
        low = action_limits[0]
        high = action_limits[1]
        space = gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
    else:
        raise NotImplementedError(type(action_limits), action_limits)

    return space

def _set_action_space():
    return convert_action_to_space({'right': COMMAND_LIMITS})

action_space = _set_action_space()['speedj']


# # Agent
# if args.env_type == "sim":
#     action_space = env.action_space
# elif args.env_type == "real":
#     action_space = env.action_space['speedj']   # check gym_custom/envs/real/ur/interface.py

agent = SAC(12, action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Load the parameter
agent.load_checkpoint("checkpoints/sac_checkpoint_{}_{}".format('single-ur3-larr-for-train-v0', num_epi), True)


def get_numpy_array():
    while True:
        try:
            user_input = input("길이가 3인 숫자를 입력하세요 (공백으로 구분): ")
            elements = user_input.split()
            
            if len(elements) != 3:
                raise ValueError("길이가 3이 아닙니다. 다시 입력해주세요.")
            
            num_array = np.array([float(element) for element in elements])
            return num_array
        except ValueError as e:
            print(e)


# Start evaluation
avg_reward = 0.
avg_step = 0.
episodes = 10
while True:
    state = env.reset()
    state = state[:12]
    print(env.goal_pos)
    episode_reward = 0
    step = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        print(action)
        next_state, reward, done, _  = env.step({
        'right': {
            'speedj': {'qd': action[:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
            'move_gripper_force': {'gf': np.array([action[6]])}
            }
        })
        if render == True :
            env.render()
        episode_reward += -np.linalg.norm(state[:3]-state[3:6])
        step += 1
        state = next_state[:12]

         # If env_type is real, evaluate just for 500 step
        if args.env_type == "real" and step == 1000:
            break   
    
    avg_reward = episode_reward/500
    print('episode_reward :', episode_reward)




