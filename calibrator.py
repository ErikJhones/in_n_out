import torch
from torch.nn import Linear
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import Parameter


class MLP(torch.nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super().__init__()
		self.lin1 = Linear(nfeat, nhid)
		self.lin2 = Linear(nhid, nclass)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.lin1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.lin2(x)
		return x

class IN_N_OUT(nn.Module):
	def __init__(self, nfeat, nclass, nhid, dropout, use_softplus=True):
		super(IN_N_OUT, self).__init__()
		self.t = None
		self.m = nn.Softplus()
		self.use_softplus = use_softplus
		
		self.scaling_model = MLP(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout)
		
	def forward(self, logits, x):
		logits2 = torch.cat((-logits, logits), dim=1)
		self.t = self.scaling_model(x)
		if self.use_softplus:
			self.t = self.m(self.t)
		self.t = self.t.mean(dim=1).unsqueeze(1)
		output = logits2 * self.t

		return output
