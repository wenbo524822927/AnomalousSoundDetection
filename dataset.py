import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import librosa
import utils
import gammatone.gtgram as gtgram


class ASDdataset(Dataset):
	def __init__(self,files):
		self.files = files

	def __len__(self):
		return len(self.files)

	def __getitem__(self,idx):
		onepath = self.files[idx]
		class_id = utils.one_hot_fan(onepath)
		audio,sr = librosa.load(onepath)
		output = gtgram.gtgram(audio,sr,0.04,0.02,64,f_min = 0)
		output = (output-np.mean(output))/np.std(output)
		output = torch.from_numpy(output)
		output = torch.unsqueeze(output,dim = 0)
		return output,class_id