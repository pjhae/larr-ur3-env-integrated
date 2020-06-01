import argparse
import numpy as np
import time

import gym_custom

def run_dual_ur3():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    for _ in range(int(60/dt)):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        env.render()
        time.sleep(dt)

if __name__ == '__main__':
    run_dual_ur3()