import torch
from torch.optim import Adam
import numpy as np
from datasets import *
from models import *
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
from netcal.metrics import ECE

import argparse
from args_models import get_args
args = get_args()

args.dataset = 'cora'

device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

seed = args.seed
dataset_name = args.dataset
gnn = args.model_gnn

torch.manual_seed(seed)
np.random.seed(seed)

data, all_edge_index = choose_dataset(dataset_name)
data = data.to(device)


model = choose_model(data, 'VGAE').to(device)

lr = args.lr
epochs = args.epochs
optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        model.eval()
        roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch+1, loss.cpu().item(), roc_auc, ap))


prefix= "test"
pos_edge_index_test = data[f'{prefix}_pos_edge_index'].to(device)
neg_edge_index_test = data[f'{prefix}_neg_edge_index'].to(device)

edge_index_test = torch.cat([pos_edge_index_test, neg_edge_index_test], dim=1)
n_test = edge_index_test.shape[1]

link_logits_test = model.predict(data.x, data.train_pos_edge_index, edge_index_test)[-n_test:]
link_logits_test = link_logits_test.unsqueeze(1)

link_labels_test = get_link_labels(pos_edge_index_test, neg_edge_index_test, device) # get link

link_labels_test = link_labels_test.type(torch.LongTensor)

evaluator20 = Evaluator(name = 'ogbl-ddi')
evaluator50 = Evaluator(name = 'ogbl-collab')
metric = ECE(15)
ece = metric.measure(torch.cat([1-link_logits_test.sigmoid(), link_logits_test.sigmoid()], dim=1).detach().numpy(), link_labels_test.numpy())
print(f'ece: {round(ece*100,2)}')
print('auc: ',round(roc_auc_score(link_labels_test.int().cpu(), link_logits_test.sigmoid().squeeze().detach().cpu())*100,2))

input_dict = {'y_pred_pos': link_logits_test.sigmoid().squeeze()[torch.where(link_labels_test == 1)[0]],
                'y_pred_neg': link_logits_test.sigmoid().squeeze()[torch.where(link_labels_test == 0)[0]]}
print('hits@20: ', round(evaluator20.eval(input_dict)['hits@20']*100,2))
print('hits@50: ', round(evaluator50.eval(input_dict)['hits@50']*100,2))

path="models/"
torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path + gnn + '_'+dataset_name+ '_seed_' + str(seed) + '.pth.tar',
    )