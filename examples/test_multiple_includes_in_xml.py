import numpy as np
import time

import gym_custom

def run_double():
    env = gym_custom.make('Practice1-motor-doublependulum-v0')
    obs = env.reset()
    dt = env.env.dt

    for _ in range(int(60/dt)):
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        qpos, qvel = obs[:env.env.model.nq], obs[-env.env.model.nv:]

        env.render()
        print('qpos: %s (degrees), qvel: %s (dps), qfrc_bias: %s'%(qpos, qvel, env.env.sim.data.qfrc_bias))
        time.sleep(dt)

if __name__ == '__main__':
    run_double()