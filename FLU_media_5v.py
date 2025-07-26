"""
FLU解码器训练函数
"""

import os
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader
from argparse import Namespace
from utils.utilities3 import *
from utils.PDE_Net import *
from utils.train_utils import *

torch.set_default_dtype(torch.float32)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set the FLU model namespace
config_FLU = Namespace(
project_name = 'FLU_3D_media_5v',
# ->
typical_case_file = r'data/typical_case_unit_media_5v_6sdf.npy',
sdf_file = r'data/media/input_data_5v.npy',
train_file = r'data/media/label_data_5v.npy',

data_x = 104,
data_y = 72,
data_z = 112,
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
online = 'online',

continue_training = False,

model = FLU_GAP,
epochs = 10000,
batch_size = 5,

lr = 0.01,
optimizer = 'AdamW',
activation = 'LeakyReLU',
loss = 'L1Loss',
hidden_channel = 64,
main_hidden_channel = 256,
weight_decay = 1e-6,
max_norm = 5,

train_ratio = 0.9,    
dropout_p = 0.1,

variables_num = 4,
sdf_input_size = 6,
sdf_layers = 3,
main_layers = 3,

latent_size = 64,

train_state_dir = 'train_state/FLU_3D_media_5v',
train_dict_dir = 'train_dict/FLU_3D_media_5v',
)

model, optimizer = train(config_FLU)