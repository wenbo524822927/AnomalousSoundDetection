import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import librosa

class ASDnetwork(nn.Module):
	def __init__(self,batchsize):
		super(ASDnetwork,self).__init__()
		self.batchsize = batchsize
		self.convblock1 = nn.Sequential(
			nn.Conv2d(1,16,3,stride=(1,2)),
			nn.BatchNorm2d(num_features = 16),
			nn.ReLU(),
			nn.Conv2d(16,32,3,stride=(1,2)),
			nn.BatchNorm2d(num_features = 32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2),stride=(1,1)))
		self.convblock2 = nn.Sequential(
			nn.Conv2d(32,48,3,stride=(1,2)),
			nn.BatchNorm2d(num_features = 48),
			nn.ReLU(),
			nn.Conv2d(48,64,3,stride=(1,1)),
			nn.BatchNorm2d(num_features = 64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2), stride=(1,1)))
		self.convblock3 = nn.Sequential(
			nn.Conv2d(64,96,3,stride=(1,2)),
			nn.BatchNorm2d(num_features = 96),
			nn.ReLU(),
			nn.Conv2d(96,128,3,stride=(1,2)),
			nn.BatchNorm2d(num_features = 128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2,2), stride=(1,1)))
		self.convblock4 = nn.Sequential(
			nn.Conv2d(128,96,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=96),
			nn.ReLU(),
			nn.Conv2d(96,64,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=64),
			nn.ReLU(),
			nn.Upsample(size = (54,58) , mode = 'bilinear',align_corners = True))
		self.convblock5 = nn.Sequential(
			nn.Conv2d(64,48,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=48),
			nn.ReLU(),
			nn.Conv2d(48,32,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=32),
			nn.ReLU(),
			nn.Upsample(size = (59,123),mode = 'bilinear',align_corners = True))
		self.convblock6 = nn.Sequential(
			nn.Conv2d(32,16,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=16),
			nn.ReLU(),
			nn.Conv2d(16,1,3,stride=(1,2)),
			nn.BatchNorm2d(num_features=1),
			nn.ReLU(),
			nn.Upsample(size = (64,499),mode = 'bilinear',align_corners = True))
		self.FC = nn.Linear(128*49*12,4)

	def encoder(self,x):
		x1 = self.convblock1(x)
		#print(x1.shape)
		x2 = self.convblock2(x1)
		#print(x2.shape)
		x3 = self.convblock3(x2)
		#print(x3.shape)
		return x3

	def decoder(self,h):
		h1 = self.convblock4(h)
		#print(h1.shape)
		h2 = self.convblock5(h1)
		#print(h2.shape)
		h3 = self.convblock6(h2)
		#print(h3.shape)
		return h3


	def forward(self,x):
		h = self.encoder(x)
		y = self.decoder(h)
		h_resize = h.resize(h.shape[0],128*49*12)
		g = self.FC(h_resize)
		return y, g
