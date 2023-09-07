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
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3_LEFT as URScriptWrapper
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no
from collections import OrderedDict
import os
import os.path as osp

# # ROS related
# import rospy
# from std_msgs.msg import String
# from geometry_msgs.msg import PoseStamped

# def listener_wait_msg():

#     rospy.init_node('ros_subscription_test_node')

#     cube_msg = rospy.wait_for_message('optitrack/cube_rainbow/poseStamped', PoseStamped)

#     return cube_msg.pose.position


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
parser.add_argument('--alpha', type=float, default=0.005, metavar='G',
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
parser.add_argument('--exp_type', default="sim",
                    help='choose sim or real')
args = parser.parse_args()

# Episode to test
num_epi = 1500

# Rendering (if exp_type is real, render should be FALSE)
render = True

# Environment
if args.exp_type == "sim":
    env = gym_custom.make('single-ur3-xy-left-front-larr-for-train-v0')
    servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}

elif args.exp_type == "real":
    env = gym_custom.make('single-ur3-larr-real-for-train-v0', # TODO
        host_ip_left='192.168.5.101',
        rate=20
    )
    servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 1, 't': 4/env.rate._freq, 'wait': False}
    # 1. Set initial as current configuration
    env.set_initial_joint_pos('current')
    env.set_initial_gripper_pos('current')
    # 2. Set inital as default configuration
    # env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0]))
    env.set_initial_joint_pos(np.array([ 1.94709782, -0.88839507,  1.76091047, -2.12800102,  0.85576529,  0.01280112]))
    env.set_initial_gripper_pos(np.array([255.0]))
    assert render is False

else:
    print("Please choose sim or real")

obs = env.reset()
dt = env.dt

if args.exp_type == "sim":
    PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':10.0}}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    # scale factor
    env.wrapper_left.ur3_scale_factor[:6] = [24.52907494 ,24.02851783 ,25.56517597, 14.51868608 ,23.78797503, 21.61325463]

elif args.exp_type == "real":
        env.env = env

if args.exp_type == "real":
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
    'movej': [np.array([-0.04, -0.04, -0.04]),
        np.array([0.04, 0.04, 0.04])] # [m]
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

action_space = _set_action_space()['movej']


agent = SAC(6, action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Load the parameter
agent.load_checkpoint("checkpoints_single/sac_checkpoint_{}_{}".format('single-ur3-larr-for-train-v0', num_epi), True)

# Constraint
class UprightConstraint(NullObjectiveBase):
    
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)

# Constraint
class FrontConstraint(NullObjectiveBase):
    
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([1, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)
    
null_obj_func_front = FrontConstraint()


# Start evaluation
avg_reward = 0.
avg_step = 0.
episodes = 10

while True:
    state = env.reset()
    state[3:6] = np.array([-0.1, -0.3, 0.8])
    state = state[:6]
    
    episode_reward = 0
    step = 0
    done = False

    while not done:

        # # # ROS related
        # cube_pos = listener_wait_msg()
        # # goal_pos = np.array([cube_pos.x, cube_pos.y, cube_pos.z]) - np.array([[-1.23483741,  0.22603086,  0.76730996]]) + np.array([0.1, -0.35, 0.9])
        # tmp = state[0:3]
        # state[0:3] = np.array([cube_pos.x, cube_pos.y, cube_pos.z]) - np.array([0.58133191, 1.26822007 ,0.75078773]) +  np.array([0.1, -0.3, 0.8])
        # state[3:6] = tmp 
        # # print(state[3:])

        action = agent.select_action(state, evaluate=True)
        q_left_des, _ ,_ ,_ = env.inverse_kinematics_ee(state[3:6]+action, null_obj_func_front, arm='left')
        dt = 1
        qvel_left = (q_left_des - env.get_obs_dict()['left']['qpos'])/dt

        next_state, reward, done, _  = env.step({
            'left': {
                'speedj': {'qd': qvel_left, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([15.0])}
            }
        })

        
        if render == True :
            env.render()
        episode_reward += reward
        step += 1
        state = next_state[:6]

         # If exp_type is real, evaluate just for 500 step
        if args.exp_type == "real" and step == 600:
            break   
    
    avg_reward = episode_reward/500
    print('episode_reward :', episode_reward)




