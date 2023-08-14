import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import itertools
import math
from joblib import Parallel, delayed
from tqdm import trange
import torch
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models import PEGConv
from torch import nn

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
    

#modified from https://github.com/shenweichen/GraphEmbedding
class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling
    
    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks
    
    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    return ("only work for DeepWalk")
        return walks
    

from gensim.models import Word2Vec
import pandas as pd

class DeepWalk:
    def __init__(self, graph, walk_length = 80, num_walks = 10, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=3, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
    

from torch_sparse import SparseTensor

def get_link_labels(pos_edge_index, neg_edge_index, device):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train_peg(model, optimizer, pos, data, x, device):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()
    
    
    link_logits = model(x, pos, data.train_pos_edge_index, neg_edge_index, data.train_pos_edge_index) # decode
    link_logits = link_logits.reshape(len(link_logits),)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index, device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.fc.weight[0][0].clamp_(1e-5,100)
    return loss

@torch.no_grad()
def test_peg(model, pos, data, x, device):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        link_logits = model(x, pos, pos_edge_index, neg_edge_index, data.train_pos_edge_index) # decode test or val
        
        link_probs = link_logits.sigmoid() # apply sigmoid
        
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, device) # get link
        
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu())) #compute roc_auc score
    return perfs


class Net(torch.nn.Module):
    def __init__(self, in_feats_dim, pos_dim, hidden_dim, node_dim):
        super(Net, self).__init__()

        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.node_dim = node_dim

        self.conv1 = PEGConv(in_channels = in_feats_dim, out_channels = hidden_dim)
        self.conv2 = PEGConv(in_channels = hidden_dim, out_channels = hidden_dim)

        self.fc = nn.Linear(2, 1)

    def embedding(self, x, pos,pos_edge_index):

        x = self.conv1(x, pos,pos_edge_index)
        x = self.conv2(x, pos,pos_edge_index)

        return x

    def forward(self, x, pos, pos_edge_index, neg_edge_index, train_pos):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x = self.embedding(x,pos, train_pos)

        nodes_first = x[edge_index[0]][:, :self.node_dim]
        nodes_second = x[edge_index[1]][:, :self.node_dim]
        pos_first = pos[edge_index[0]]
        pos_second = pos[edge_index[1]]

        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)

        pred = (nodes_first * nodes_second).sum(dim=-1)  # dot product
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))

        return out