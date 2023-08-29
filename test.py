import numpy as np
from collections import OrderedDict
import gym_custom
from gym_custom import spaces
import torch

log_alpha = torch.zeros(1, requires_grad=True)
log_alpha = log_alpha*0.1

print(log_alpha)
