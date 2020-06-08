import argparse
import numpy as np
import time

import gym_custom
from gym_custom.envs.custom.dual_ur3_env import URScriptWrapper, NullObjectiveBase

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
    R_base, p_base, T_base = env.forward_kinematics(body_name='right_arm_rotz')
    # R_hand, p_hand, T_hand = env.forward_kinematics(body_name='right_gripper:hand')
    R_hand, p_hand, T_hand = env.forward_kinematics(body_name='right_ee_link')
    print('base:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[0,:], p_base))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[0,:,:], R_base))
    print('hand:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[-1,:], p_hand))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[-1,:,:], R_hand))

def servoj_and_forceg(env_type='sim'):
    assert env_type in ['sim', 'real']

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.8])
    ee_pos_left = np.array([-0.1, -0.5, 0.8])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    qpos_des = env.init_qpos.copy()
    qpos_des[0:env.ur3_nqpos] = q_right_des
    qpos_des[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des
    env.render()
    time.sleep(5.0)
    while True:
        env.set_state(qpos_des, env.init_qvel)
        env.render()

    PID_gains = {'P': 1.0, 'I': 0.5, 'D': 0.2}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])*0.01
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*180.0/np.pi and  qvel > 1e-2*180.0/np.pi:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([1.0, 1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        time.sleep(1*dt)

def speedj_and_forceg(env_type='sim'):
    assert env_type in ['sim', 'real']

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    PI_gains = {'P': 0.20, 'I': 5.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)

    while qvel_err > 1e-2*180.0/np.pi:
        pass

def pick_and_place(env_type='sim'):
    assert env_type in ['sim', 'real']

    pass

if __name__ == '__main__':
    # show_dual_ur3()
    # run_dual_ur3()
    # test_fkine_ikine()
    servoj_and_forceg()
    # speedj_and_forceg()
