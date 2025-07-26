"""
CAE自动编码器训练函数，用于流场数据低维特征提取
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


config_CAE = Namespace(
project_name = 'CAE_3D_media_5v',
# ->
train_file = r'data/media/label_data_5v.npy',

data_x = 104,
data_y = 72,
data_z = 112,
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
online = 'online',

continue_training = False,

model = CAE_GAP,
epochs = 10000,
batch_size = 10,
lr = 0.001,
optimizer = 'AdamW',
activation = 'LeakyReLU',
loss = 'L1Loss',
train_ratio = 0,
dropout_p = 0,

input_size = 4,	
layers = 3,
latent_size = 64,
hidden_channel = 64,

weight_decay = 1e-5,
max_norm = 5,

train_state_dir = 'train_state/CAE_3D_media_5v',
train_dict_dir = 'train_dict/CAE_3D_media_5v',
)

model, optimizer = train(config_CAE)
