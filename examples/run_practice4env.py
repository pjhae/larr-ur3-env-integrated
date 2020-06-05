import numpy as np
import time

import gym_custom

def forward_kinematics_DH(q, kinematic_params):
    assert len(q) == 6

    T_0_i = kinematic_params['T_base']

    T = np.zeros([7, 4, 4])
    R = np.zeros([7, 3, 3])
    p = np.zeros([7, 3])
    # Base frame
    T[0,:,:] = T_0_i
    R[0,:,:] = T_0_i[0:3,0:3]
    p[0,:] = T_0_i[0:3,3]

    for i in range(6):
        ct = np.cos(q[i] + kinematic_params['offset'][i])
        st = np.sin(q[i] + kinematic_params['offset'][i])
        ca = np.cos(kinematic_params['alpha'][i])
        sa = np.sin(kinematic_params['alpha'][i])

        T_i_iplus1 = np.array([[ct, -st*ca, st*sa, kinematic_params['a'][i]*ct],
                                [st, ct*ca, -ct*sa, kinematic_params['a'][i]*st],
                                [0, sa, ca, kinematic_params['d'][i]],
                                [0, 0, 0, 1]])
        T_0_i = np.matmul(T_0_i, T_i_iplus1)
        # cf. base frame at i=0
        T[i+1, :, :] = T_0_i
        R[i+1, :, :] = T_0_i[0:3,0:3]
        p[i+1, :] = T_0_i[0:3,3]

    return R, p, T

def test_kinematics():
    q = np.zeros([6])
    kinematic_params = {}
    kinematic_params['T_base'] = np.eye(4)
    kinematic_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12*0]) # in m
    kinematic_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
    kinematic_params['alpha'] = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
    kinematic_params['offset'] = np.zeros([6])
    R, p, T = forward_kinematics_DH(q, kinematic_params)

    env = gym_custom.make('Practice4-v0')
    joint6_idx = env.model.body_names.index('joint6')
    T_mujoco = np.eye(4)
    T_mujoco[0:3,0:3] = env.sim.data.body_xmat[joint6_idx].reshape([3,3])
    T_mujoco[0:3,3] = env.sim.data.body_xpos[joint6_idx]

    print('DH pos: %s'%(p[-1,:]))
    print('mujoco pos: %s'%(T_mujoco[0:3,3]))

def run_dynamics():
    env = gym_custom.make('Practice4-v0')
    env.reset()
    dt = env.env.dt

    while True:
        env.render()
        time.sleep(dt)
    
    while True:
        obs, _, _, _ = env.step(np.array([-5]))
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]
        env.render()
        print('qpos: %s (degrees), qvel: %s (dps), qfrc_bias: %s'%(qpos, qvel, env.env.sim.data.qfrc_bias))
        time.sleep(dt)

    for _ in range(int(120/dt)):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]

        env.render()
        print('qpos: %s (degrees), qvel: %s (dps), qfrc_bias: %s'%(qpos, qvel, env.env.sim.data.qfrc_bias))
        time.sleep(dt*10)

if __name__ == '__main__':
    run_dynamics()
    # test_kinematics()
