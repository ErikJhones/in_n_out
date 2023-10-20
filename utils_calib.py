import math
import numpy as np
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

def get_interventional_emb(train_edges, probe_edge_index, model, x, device, type_set='test'):
    '''
    make sure that the positive edges are in the first half of the vector, 
    while the negative ones are in the second half.
    '''
    
    
    size_vector = probe_edge_index.shape[1]
    
    for i, j in enumerate(probe_edge_index.T):
        jprime = j
        jprime[0], jprime[1]= j[1], j[0]
        if type_set == 'train' and i <= size_vector//2:
            # here, probe_edge_index are equal to train_edges. so, there isn't problem to remove
            # the edge to probe_edge_index and assign to edge_index
            edge_index = torch.cat((probe_edge_index[:, :i], probe_edge_index[:, i+1:]), dim=1)
        else:
            edge_index = torch.cat([train_edges, j[:, None], jprime[:, None]], dim=1)
        node_embeddings = model.embedding(x, edge_index)
        nodes_first_ = node_embeddings[j[0]]
        nodes_second_ = node_embeddings[j[1]]
        
        yield nodes_first_ * nodes_second_

def agr_emb(edge_index, embedding_edges):

    nodes_first = embedding_edges[edge_index[0]]
    nodes_second = embedding_edges[edge_index[1]]
    return nodes_first * nodes_second
    
def accuracy(pred, targ):
    pred = torch.softmax(pred, dim=1)
    pred_max_index = torch.max(pred, 1)[1]
    ac = ((pred_max_index == targ).float()).sum().item() / targ.size()[0]
    return ac

class ECELoss(nn.Module):

    def __init__(self, n_bins=20, device='cpu'):
        """
        n_bins (int): number of confidence interval bins
        """
        self.device = device
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(self.device)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1).to(self.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def intra_distance_loss(output, labels):
    output = torch.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = torch.where(pred_max_index==labels)
    incorrect_i = torch.where(pred_max_index!=labels)
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    loss = (torch.sum(1 - pred[correct_i] + sub_pred[correct_i]) + torch.sum(pred[incorrect_i]-sub_pred[incorrect_i])) / labels.size()[0]
    return loss
