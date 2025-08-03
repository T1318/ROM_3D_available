import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================================================================
# ====================================================================================================

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

# ======================================================================================#

class MLP(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels, activation = nn.Identity()):
		super(MLP, self).__init__()
		self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
		self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)
		self.func = activation

	def forward(self, x):
		x = self.mlp1(x)
		x = self.func(x)
		x = self.mlp2(x)
		return x

# ====================================GAP======================================= #
# 定义编码器
class Encoder_GAP(nn.Module):
	def __init__(self, config, input_size, hidden_channel, output_size, layers, activation, dropout_p):
		super().__init__()
		self.config = config

		input_channel = input_size
		encoder_cnn = []
		for i in range(layers):
			if i == layers-1:
				encoder_cnn.append(nn.Conv3d(hidden_channel, output_size, 3, stride=1, padding=1))
				encoder_cnn.append(nn.BatchNorm3d(output_size))
				encoder_cnn.append(activation())
				encoder_cnn.append(nn.MaxPool3d(2, stride=2))

			else:
				encoder_cnn.append(nn.Conv3d(input_channel, hidden_channel, 3, stride=1, padding=1))
				encoder_cnn.append(nn.BatchNorm3d(hidden_channel))
				encoder_cnn.append(activation())
				encoder_cnn.append(nn.MaxPool3d(2, stride=2))
				input_channel = hidden_channel

		self.encoder_cnn = nn.Sequential(*encoder_cnn)

	def forward(self, x):
		x = self.encoder_cnn(x)
		return x

# 定义解码器
class Decoder_GAP(nn.Module):
	def __init__(self, config, input_size, hidden_channel, output_size, layers, activation, dropout_p):
		super().__init__()
		
		linear2_inlet = int(input_size*((config.data_x/2**layers) * (config.data_y/2**layers) * (config.data_z/2**layers)))
		linear1_inlet = int(input_size)

		self.decoder_lin = nn.Sequential(
			nn.Linear(linear1_inlet, linear2_inlet),
			nn.BatchNorm1d(linear2_inlet),
			activation(),
		)

		self.unflatten = nn.Unflatten(dim=1,unflattened_size=(input_size, int(config.data_x/2**layers), int(config.data_y/2**layers), int(config.data_z/2**layers)))

		input_channel = input_size
		decoder_cnn = []
		for i in range(layers-1):
			# 添加解卷积层
			decoder_cnn.append(nn.ConvTranspose3d(input_channel, hidden_channel, 3, stride=2, padding=1, output_padding=1))
			decoder_cnn.append(nn.Conv3d(hidden_channel, hidden_channel, 3, stride=1, padding=1))
			decoder_cnn.append(nn.BatchNorm3d(hidden_channel))
			decoder_cnn.append(activation())
			input_channel = hidden_channel
		# 添加最后的输出层
		decoder_cnn.append(nn.ConvTranspose3d(hidden_channel, output_size, 3, stride=2, padding=1, output_padding=1))
		decoder_cnn.append(nn.Conv3d(output_size, output_size, 3, stride=1, padding=1))
		
		self.decoder_conv = nn.Sequential(*decoder_cnn)

	def forward(self, x):
		x = self.decoder_conv(x)
		return x

# ============================================================================== #

# 定义卷积自动编码器_GAP
class CAE_GAP(nn.Module):
	def __init__(self,config):
			super(CAE_GAP, self).__init__()
			self.config = config
			self.activation = torch.nn.__dict__[self.config.activation]
			self.encoder = Encoder_GAP(self.config, self.config.input_size, self.config.hidden_channel, self.config.latent_size, self.config.layers, self.activation, self.config.dropout_p)
			self.decoder = Decoder_GAP(self.config, self.config.latent_size, self.config.hidden_channel, self.config.input_size, self.config.layers, self.activation, self.config.dropout_p)

			self.MLP = MLP(self.config.latent_size, self.config.latent_size, self.config.latent_size)

	def forward(self, x):
			x = self.encoder(x)
			x = self.MLP(x)
			x = self.decoder(x)
			return x
	
# 定义FLU_GAP
class FLU_GAP(nn.Module):
	def __init__(self,config):
		super(FLU_GAP, self).__init__()
		self.config = config
		self.activation = torch.nn.__dict__[self.config.activation]
		self.SDF_net = Encoder_GAP(self.config, self.config.sdf_input_size, self.config.hidden_channel, self.config.latent_size, self.config.sdf_layers, self.activation, self.config.dropout_p)
		self.main_net = Decoder_GAP(self.config, self.config.latent_size, self.config.main_hidden_channel, self.config.variables_num, self.config.main_layers, self.activation, self.config.dropout_p)
		self.MLP = MLP(self.config.latent_size, self.config.latent_size, self.config.latent_size)

	def forward(self, typical_case, sdf):
		self.laten_fluid = self.MLP(typical_case)
		self.laten_fluid = torch.sum(self.laten_fluid, axis=0, keepdims=True)
		self.laten_sdf = self.SDF_net(sdf)

		self.laten = self.laten_sdf * self.laten_fluid

		self.latent = self.MLP(self.laten)
		self.out = self.main_net(self.laten)
		return self.out
	