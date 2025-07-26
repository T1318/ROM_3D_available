import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, OneCycleLR
from argparse import Namespace
from utils.utilities3 import *
from utils.PDE_Net import *
from utils.plot_utils import *

torch.set_default_dtype(torch.float32)

def creat_dataloader(config):
	if 'CAE' in config.project_name:
		label = torch.from_numpy(np.load(config.train_file))
		label_norm = UnitGaussianNormalizer(label)
		label = label_norm.encode(label)

		if config.train_ratio:
			train_size = int(len(label) * config.train_ratio)
			train_lable = label[:train_size]
			test_label = label[train_size:]

			train_loader = DataLoader(torch.utils.data.TensorDataset(train_lable), batch_size = config.batch_size, shuffle = True, drop_last = False)
			test_loader = DataLoader(torch.utils.data.TensorDataset(test_label), batch_size = config.batch_size, shuffle = True, drop_last = False)
			return train_loader, test_loader

		else:
			train_label = label

			train_loader = DataLoader(torch.utils.data.TensorDataset(train_label), batch_size = config.batch_size, shuffle = True, drop_last = False)

			return train_loader
	
	#========================================#

	elif 'FLU' in config.project_name:
		sdf = torch.from_numpy(np.load(config.sdf_file)).float()
		label = torch.from_numpy(np.load(config.train_file)).float()
		label_norm = UnitGaussianNormalizer(label)
		label = label_norm.encode(label)

		if config.train_ratio:
			train_size = int(len(label) * config.train_ratio)

			train_sdf = sdf[:train_size]
			test_sdf = sdf[train_size:]
			train_lable = label[:train_size]
			test_label = label[train_size:]

			train_loader = DataLoader(torch.utils.data.TensorDataset(train_sdf, train_lable), batch_size = config.batch_size, shuffle = True, drop_last = False)
			test_loader = DataLoader(torch.utils.data.TensorDataset(test_sdf, test_label), batch_size = config.batch_size, shuffle = True, drop_last = False)

			return train_loader, test_loader
		
		else:
			train_sdf = sdf
			train_label = label

			train_loader = DataLoader(torch.utils.data.TensorDataset(train_sdf, train_label), batch_size = config.batch_size, shuffle = True, drop_last = False)
			return train_loader


def train_epoch(config, model, optimizer, loss, train_loader):
	scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=config.epochs)
	if 'CAE' in config.project_name:

		model.train()
		train_loss_per_epoch = 0
		for batch in train_loader:
			loss_per_batch = 0
			x = batch[0]

			x = x.float().to(config.device)

			pred = model(x)

			loss_per_batch = loss(pred.clone(), x.clone())

			regularization_loss = 0
			for param in model.parameters():
				regularization_loss += torch.norm(param, p=2)
			loss_per_batch += config.weight_decay * regularization_loss

			train_loss_per_epoch += loss_per_batch.item()

			optimizer.zero_grad()
			loss_per_batch.backward()
			optimizer.step()
			scheduler.step()

		train_loss_per_epoch /= len(train_loader)
	#========================================#
	
	elif 'FLU' in config.project_name:
		model.train()
		train_loss_per_epoch = 0
		typical_case = torch.from_numpy(np.load(config.typical_case_file)).float()
		for batch in train_loader:
			loss_per_batch = 0
			sdf, label = batch

			typical_case = typical_case.float().to(config.device)
			sdf = sdf.float().to(config.device)
			label = label.float().to(config.device)

			pred = model(typical_case, sdf)

			loss_per_batch = loss(pred.clone(), label.clone())
			regularization_loss = 0
			for param in model.parameters():
				regularization_loss += torch.norm(param, p=2)
			loss_per_batch += config.weight_decay * regularization_loss

			train_loss_per_epoch += loss_per_batch.item()

			optimizer.zero_grad()
			loss_per_batch.backward()
			optimizer.step()
			scheduler.step()

		train_loss_per_epoch /= len(train_loader)
	return train_loss_per_epoch

def train(config):
	model = config.model(config).to(config.device)
	model.apply(weight_init)

	optimizer = torch.optim.__dict__[config.optimizer](params=model.parameters(), lr=config.lr)
	loss = getattr(nn, config.loss)()

	if config.continue_training:
		load_model(config.train_state_dir, model, optimizer)
		
	#================
	now_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
	#================

	if not os.path.exists(config.train_dict_dir):
			os.makedirs(config.train_dict_dir)
	if not os.path.exists(config.train_state_dir):
			os.makedirs(config.train_state_dir)
			
	#================

	dict_path = os.path.join(config.train_dict_dir, '{}_{}.json'.format(config.project_name, now_time))
	save_dict(dict_path, config)

	for epoch in range(config.epochs+1):
		time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
		train_loader = creat_dataloader(config)	# [Batch,variables,x,y,z]
		train_loss_per_epoch = train_epoch(config, model, optimizer, loss, train_loader)
		print('{}, Epoch: {}, Train Loss: {:.4f}'.format(time, epoch, train_loss_per_epoch))
		if epoch % 1000 == 0:
			save_path = os.path.join(config.train_state_dir, '{}_{}.pth'.format(config.project_name, now_time))
			save_model(save_path, epoch, model, optimizer, now_time)

	return model, optimizer

