import os
import json
from argparse import Namespace
import torch
import numpy as np
# import vtkmodules.all as vtk
import operator
from functools import reduce
from scipy.interpolate import griddata

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
	def __init__(self, x, eps=0.00001):
			super(UnitGaussianNormalizer, self).__init__()

			# x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
			self.mean = torch.mean(x, 0)
			self.std = torch.std(x, 0)
			self.eps = eps

	def encode(self, x):
			x = (x - self.mean) / (self.std + self.eps)
			return x

	def decode(self, x, sample_idx=None):
			if sample_idx is None:
					std = self.std + self.eps # n
					mean = self.mean
			else:
					if len(self.mean.shape) == len(sample_idx[0].shape):
							std = self.std[sample_idx] + self.eps  # batch*n
							mean = self.mean[sample_idx]
					if len(self.mean.shape) > len(sample_idx[0].shape):
							std = self.std[:,sample_idx]+ self.eps # T*batch*n
							mean = self.mean[:,sample_idx]

			# x is in shape of batch*n or T*batch*n
			x = (x * std) + mean
			return x

	def cuda(self):
			self.mean = self.mean.cuda()
			self.std = self.std.cuda()

	def cpu(self):
			self.mean = self.mean.cpu()
			self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
	def __init__(self, x, eps=0.00001):
			super(GaussianNormalizer, self).__init__()

			self.mean = torch.mean(x)
			self.std = torch.std(x)
			self.eps = eps

	def encode(self, x):
			x = (x - self.mean) / (self.std + self.eps)
			return x

	def decode(self, x, sample_idx=None):
			x = (x * (self.std + self.eps)) + self.mean
			return x

	def cuda(self):
			self.mean = self.mean.cuda()
			self.std = self.std.cuda()

	def cpu(self):
			self.mean = self.mean.cpu()
			self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
	def __init__(self, x, low=0.0, high=1.0):
			super(RangeNormalizer, self).__init__()
			mymin = torch.min(x, 0)[0].view(-1)
			mymax = torch.max(x, 0)[0].view(-1)

			self.a = (high - low)/(mymax - mymin)
			self.b = -self.a*mymax + high

	def encode(self, x):
			s = x.size()
			x = x.view(s[0], -1)
			x = self.a*x + self.b
			x = x.view(s)
			return x

	def decode(self, x):
			s = x.size()
			x = x.view(s[0], -1)
			x = (x - self.b)/self.a
			x = x.view(s)
			return x



# print the number of parameters
def count_params(model):
	c = 0
	for p in list(model.parameters()):
		c += reduce(operator.mul, list(p.size()))
	return c

# save the model
def save_model(save_path, epoch, model, optimizer, now_time):
	torch.save({'epoch': epoch, 'now_time': now_time, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

# load the model
def load_model(save_path, model, optimizer = None, file_path = None):
	if file_path is None:
		file_list = os.listdir(save_path)
		file_list.sort(key = lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
		last_file = os.path.join(save_path, file_list[-1])
	else:
		last_file = os.path.join(save_path, file_path)

	model_data = torch.load(last_file)
	model.load_state_dict(model_data['model_state_dict'])
	if optimizer is not None:
		optimizer.load_state_dict(model_data['optimizer_state_dict'])
	print('Load model at time:',model_data['now_time'])

# save the dict
def save_dict(save_path, config):
	config.device = str(config.device)
	config.model = str(config.model)
	if hasattr(config,'encoder'):
		config.encoder = str(config.encoder)
	if hasattr(config,'mlp'):
		config.mlp = str(config.mlp)
	config_dict = vars(config)

	with open(save_path, 'w') as f:
		json.dump(config_dict, f, indent=2)

# load the dict
def load_dict(save_path, file_path = None):
	if file_path is None:
		file_list = os.listdir(save_path)
		file_list.sort(key = lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
		last_file = os.path.join(save_path, file_list[-1])
	else:
		last_file = os.path.join(save_path, file_path)

	with open(last_file, 'r') as f:
		config_dict = json.load(f)
	config = Namespace(**config_dict)
	print('Load model at time:',last_file.split('.')[0])
	return config

# 0-1 normalization
def normalize(data):
	'''
	min:	[batch_size,variable,x,y,z],		
	max:	[batch_size,variable,x,y,z],		
	'''
	min = torch.zeros((data.shape[0],data.shape[1]))
	max = torch.zeros((data.shape[0],data.shape[1])) 
	data_norm = torch.zeros(data.shape)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			min[i,j] = torch.min(data[i,j])
			max[i,j] = torch.max(data[i,j])
			data_norm[i,j] = (data[i,j]-min[i,j])/(max[i,j]-min[i,j]+1e-6)
	return data_norm,min,max

# inverse 0-1 normalization
def inverse_normalize(data,min,max):
	# data = torch.zeros(data.shape)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i,j] = data[i,j]*(max[i,j]-min[i,j]+1e-6)+min[i,j]
	return data

# uniform downsample
def uniformDownSample(data, downSamplePositon, method='linear'):
		'''
		data: [x, y, u, v, p]
		downSamplePositon: [x, y]
		'''
		interpolated_uvp = griddata(data[:,0:2], data[:,2:5], downSamplePositon, method=method)
		return interpolated_uvp

def cross_fit(repeat_times, matrix1, matrix2 = None):
	if matrix2 != None and repeat_times == 2:
		result = torch.empty_like(matrix1.repeat(2,1))
		result[0::2] = matrix1
		result[1::2] = matrix2
	
	else:
		result = torch.empty_like(matrix1.repeat(repeat_times,1))
		for i in range(repeat_times):
			result[i::repeat_times] = matrix1

	return result