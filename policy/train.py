"""
    This file carries out training of the specified agent on the specified environment.
"""

import warnings
import os

from pathlib import Path

import hydra
import numpy as np
import torch

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True