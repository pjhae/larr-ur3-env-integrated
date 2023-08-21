import argparse
import datetime
import gym_custom
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3 as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from collections import OrderedDict
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory, HERMemory
from utils import VideoRecorder

## parser 와 train, test 및 파라미터 자동저장

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
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
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
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym_custom.make('dual-ur3-pick-and-place-larr-for-train-v0')
servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
gripper_scale_factor = np.array([1.0])
env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)

# Max episode
max_episode_steps = 1000

# For reproducibility
env.seed(args.seed)
env.action_space.seed(args.seed)   
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# For video
video_directory = '/home/jonghae/larr-ur3-env/sac_dual/video/{}'.format(datetime.datetime.now().strftime("%H:%M:%S %p"))
video = VideoRecorder(dir_name = video_directory)


COMMAND_LIMITS = {
    'speedj': [np.array([-np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])*0.1,
        np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi, np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])*0.1], # [rad/s]
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

# # Set motor gain scale
env.wrapper_right.ur3_scale_factor[:6] = [24.52907494, 24.02851783, 25.56517597, 14.51868608, 23.78797503, 21.61325463]
env.wrapper_left.ur3_scale_factor[:6] =  [23.03403947, 23.80201627, 30.65127641, 14.93660589, 23.06927071, 21.52280244]

# Agent
agent = SAC(36, action_space, args)

# Tesnorboard
writer = SummaryWriter('runs_dual/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'dual-ur3-pick-and-place-larr-for-train-v0',
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
# (HER) HER_memory = HERMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    state = state[:36]
    while not done:
        if args.start_steps > total_numsteps:
            action = action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _  = env.step({
            'right': {
                'speedj': {'qd': action[:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': action[6:12], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            }
        })
        
        # next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        env.render()
        # Ignore the "done" signal if it comes from hitting the time horizon. (max timestep 되었다고 done 해서 next Q = 0 되는 것 방지)
        mask = 1 if episode_steps == max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state[:36], mask) # Append transition to memory

        state = next_state[:36]
        
    if total_numsteps > args.num_steps:
        break   
    
    # ## 1
    # HER_batch = HER_memory.sample(state)
    # HER_batch_length = len(HER_batch)
    # HER_memory.clear()
    # memory.push_batch(HER_batch, HER_batch_length)

    ## 2
    # print("T :",state[3:6])
    # (HER) state, action, reward, next_state, done  = HER_memory.sample(state)
    # (HER) for i in range(len(state)):
    # (HER)     memory.push(state[i], action[i], reward[i], next_state[i], done[i])


    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    if i_episode % 10 == 0:
        agent.save_checkpoint('dual-ur3-pick-and-place-larr-for-train-v0',"{}".format(i_episode))

    if i_episode % 20 == 0 and args.eval is True:
        video.init(enabled=True)
        avg_reward = 0.
        avg_step = 0.
        episodes = 5
        for _  in range(episodes):
            state = env.reset()
            state = state[:36]
            episode_steps = 0
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                video.record(env.render(mode='rgb_array', camera_id=1))
                next_state, reward, done, _  = env.step({
                    'right': {
                        'speedj': {'qd': action[:6], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                        'move_gripper_force': {'gf': np.array([10.0])}
                    },
                    'left': {
                        'speedj': {'qd': action[6:12], 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                        'move_gripper_force': {'gf': np.array([10.0])}
                    }
                })
                episode_reward += -np.linalg.norm(state[:6]-state[6:12])
                episode_steps += 1

                state = next_state[:36]
            avg_reward += episode_reward
            avg_step += episode_steps
        avg_reward /= episodes
        avg_step /= episodes

        video.save('test_{}.mp4'.format(i_episode))
        video.init(enabled=False)

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. step: {}".format(episodes, round(avg_reward, 2), round(avg_step, 2)))
        print("----------------------------------------")

env.close()


