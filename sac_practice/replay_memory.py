import random
import numpy as np
import os
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch, length):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position:self.position + length] = batch
        self.position = (self.position + length) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class HERMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # def sample(self, terminal_state):
    #     batch = random.sample(self.buffer,len(self.buffer))
    #     state, action, reward, next_state, done = map(np.stack, zip(*batch))
    #     # Change to HER transition
    #     for i in range(len(self.buffer)):
    #         state[i][:3] = terminal_state[:3]
    #         next_state[i][:3] = terminal_state[:3]
    #         if np.linalg.norm(state[i][:3] - state[i][3:6]) < 0.03:
    #             reward[i] = 1
    #         else:
    #             reward[i] = 0
     
    #     return state, action, reward, next_state, done 
    
    def sample(self, terminal_state):
        batch = self.buffer
        # Change to HER transition
        for i in range(len(self.buffer)):
            state, action, reward, next_state, done = batch[i]
            state[:3] = terminal_state[:3]
            next_state[:3] = terminal_state[:3]
            if np.linalg.norm(state[:3] - state[3:6]) < 0.03:
                reward = 1
            else:
                reward = 0
            batch[i] = state, action, reward, next_state, done 
    
        return batch

    def clear(self):
        self.buffer = []
        self.position = 0
