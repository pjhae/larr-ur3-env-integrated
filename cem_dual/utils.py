import math
import torch
import os
import sys

import imageio
import numpy as np


################ CEM related ####################

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def save_data(data, file_name, data_dir = "cem_dual/data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, file_name)
    np.save(file_path, data)
    print("SAVE!")

def load_data(file_path):
    loaded_data = np.load(file_path)
    return loaded_data
