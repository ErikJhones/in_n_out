from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import torch_geometric.transforms as T
from copy import deepcopy
import torch_geometric.utils as U
from torch_geometric.data.data import Data
from torch_geometric.datasets import Twitch
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
import os
from torch import manual_seed
from numpy import random
manual_seed(42)
random.seed(42)
def preprocess_data(data, edge_split):
	
    data = data.clone()
    data = T.ToSparseTensor()(data)
    data = data.coalesce()
    data.adj_t = data.adj_t.float()
    row, col, data.edge_weight = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)
    del data.edge_year

    edge_split = deepcopy(edge_split)
    edge_index, edge_weight = U.coalesce(edge_split['train']['edge'].t(),
                                         edge_split['train']['weight'])
    edge_split['train']['edge'] = edge_index.t()
    edge_split['train']['weight'] = edge_weight
    del edge_split['train']['year']
    del edge_split['valid']['year']
    del edge_split['test']['year']

    return data, edge_split

def choose_dataset(dataset_name='cora'):

    os.makedirs("datasets", exist_ok=True)
    
    if 'ogbl' in dataset_name:
        if dataset_name in ['ogbl-vessel', 'ogbl-ddi']:
        
            dataset = PygLinkPropPredDataset(dataset_name, root='datasets')
            data = dataset[0]
            all_edge_index = data.edge_index
            edge_split = dataset.get_edge_split()
            emb = torch.nn.Embedding(data.num_nodes, 128)
            feat = emb.weight
            data = Data(x=feat, 
                        val_pos_edge_index=edge_split['valid']['edge'].T,
                        test_pos_edge_index=edge_split['test']['edge'].T,
                        train_pos_edge_index=edge_split['train']['edge'].T,
                        val_neg_edge_index=edge_split['valid']['edge_neg'].T,
                        test_neg_edge_index=edge_split['test']['edge_neg'].T
                        )

        elif dataset_name == 'ogbl-collab':

            dataset = PygLinkPropPredDataset(dataset_name, root='datasets')
            evaluator = Evaluator(dataset_name)
            data = dataset[0]
            all_edge_index = data.edge_index
            edge_split = dataset.get_edge_split()
            data, edge_split = preprocess_data(data, edge_split)
            train_pos_edges = edge_split['train']['edge']
            val_pos_edges = edge_split['valid']['edge']
            val_neg_edges = edge_split['valid']['edge_neg']
            test_pos_edges = edge_split['test']['edge']
            test_neg_edges = edge_split['test']['edge_neg']
            data = Data(x=data.x,
                        val_pos_edge_index=val_pos_edges.T,
                        test_pos_edge_index=test_pos_edges.T,
                        train_pos_edge_index=train_pos_edges.T,
                        val_neg_edge_index=val_neg_edges.T,
                        test_neg_edge_index=test_neg_edges.T
                       )
        else:
            raise Exception('Just obgl ddi, collab and vessel')
            
    else:
        if dataset_name in ['RU', 'PT']:
            dataset = Twitch("datasets", dataset_name, transform=T.NormalizeFeatures())
            
        elif dataset_name == 'chameleon':
            dataset = WikipediaNetwork("datasets", dataset_name, transform=T.NormalizeFeatures())

        elif dataset_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid("datasets", dataset_name, transform=T.NormalizeFeatures())
            
        else:
            raise Exception('Just cora, citeseer, pubmed, RU, PT or chameleon')
            
        data = dataset[0]
        all_edge_index = data.edge_index
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, 0.05, 0.1)

    return data, all_edge_index

