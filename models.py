import torch
import torch.nn as nn
from torch import manual_seed
from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
import torch.nn.functional as F
from utils import get_link_labels
import random
random.seed(42)
manual_seed(42)


def get_model_data(model, data, prefix, device):
    model.eval()
    
    pos_edge_index = data[f'{prefix}_pos_edge_index']
    if prefix == 'train':
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index, #positive edges
            num_nodes=data.num_nodes, # number of nodes
            num_neg_samples=data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
    else:
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        
    edge_index_ = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    n_ = edge_index_.shape[1]
    
    link_logits = model.predict(data.x, data.train_pos_edge_index, edge_index_)[-n_:]
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)
    link_labels = link_labels.type(torch.LongTensor)
    
    return edge_index_, link_logits, link_labels, n_


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels):
        super(DeepVGAE, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def embedding(self, x, pos_edge_index):
        return self.encode(x, pos_edge_index)

    def predict(self, x, pos_edge_index, neg_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=False)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=False)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        return pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
        
        
def choose_model(data, model_name='VGAE'):
	if model_name == 'VGAE':
		enc_in_channels = data.x.shape[1]
		enc_hidden_channels = 32
		enc_out_channels = 16
		model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels)
	
	elif model_name == 'GraphSAGE':
		raise Exception('Model not implemented')
	elif model_name == 'PEG':
		raise Exception('Model not implemented')
	else:
		raise Exception('Model not implemented')

	return model
