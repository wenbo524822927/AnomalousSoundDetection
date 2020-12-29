import numpy as np
import torch

#fan_id has four kinds: 00 02 04 06
def one_hot_fan(onepath):
	mac_id = int(onepath[-14:-13])
	output = torch.tensor([0.,0.,0.,0.])
	if mac_id == 0 :
		output = torch.tensor([1.,0.,0.,0.])
	elif mac_id == 2 :
		output = torch.tensor([0.,1.,0.,0.])
	elif mac_id == 4 :
		output  = torch.tensor([0.,0.,1.,0.])
	elif mac_id == 6 :
		output = torch.tensor([0.,0.,0.,1.])
	return output
