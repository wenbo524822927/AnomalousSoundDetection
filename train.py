import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import librosa
from network import ASDnetwork as network
from dataset import ASDdataset as dataset

#initial setting
datapath = '/home/bowen/study/ASD/data/dev_data_fan/train/'
files = librosa.util.find_files(datapath,ext = 'wav')
files_Length = len(files)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.is_available())
#network setting
batchsize = 16

#train_set/validate_set/test_set
train_dataset = dataset(files)
#train_loader/validate_loader/test_loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=False)


#model
model = network(batchsize).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
#training
num_epochs = 20
total_step = len(train_loader)
for epoch in range(num_epochs):
#running_loss = 0.0
	for i,(gtg,classes) in enumerate(train_loader):
		gtg = gtg.float()
		gtg = gtg.to(device)
		#classes = classes.long()
		classes = classes.to(device)
		output1,output2 = model(gtg)
		loss1 = criterion1(output1,gtg)
		loss2 = criterion2(output2,torch.argmax(classes, dim=1))
		loss = loss1
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#print statistics
		print ('Epoch [{}/{}], Step [{}/{}], Loss1: {:.4f}, Loss2: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss1.item(), loss2.item()))

#save trained model
PATH = './fan_2d_net_1_00.pth'
torch.save(model.state_dict(),PATH)
