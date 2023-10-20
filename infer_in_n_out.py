import torch
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
from calibrator import IN_N_OUT
import argparse
from args_calib import get_args
from datasets import *
import numpy as np
from models import *
from utils_calib import *

device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Arguments
args = get_args()
# args.type_process_emb = 'sub'
# args.dataset = 'cora'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

data, _ = choose_dataset(args.dataset)
data = data.to(device)

path = './models/'

model = choose_model(data, args.model_gnn).to(device)
    
ckpt_destilado = torch.load(path + args.model_gnn + '_'+args.dataset+ '_seed_' + str(seed) + '.pth.tar', map_location=torch.device(device))
model.load_state_dict(ckpt_destilado["model_state"]) 

for para in model.parameters():
    para.requires_grad = False

edge_index_test, link_logits_test, link_labels_test, n_test = get_model_data(model, data, 'test', device)
node_embeddings = model.embedding(data.x, data.train_pos_edge_index)
link_emb_test = agr_emb(edge_index_test, node_embeddings)
intervential_emb_test = torch.load('./inter_emb_model/intervential_emb_test_'+args.dataset+'_'+args.model_gnn+'.pt')

if args.type_process_emb == 'sub':
    interv_link_emb_test = (intervential_emb_test - link_emb_test)

elif args.type_process_emb == 'dist':
    interv_link_emb_test = torch.pairwise_distance(intervential_emb_test, link_emb_test, p=1).unsqueeze(1)

elif args.type_process_emb == 'sub_signed':
    interv_link_emb_test = (intervential_emb_test - link_emb_test) * link_logits_test.sign()#.unsqueeze(1)

elif args.type_process_emb == 'dist_signed':
    interv_link_emb_test = (intervential_emb_test - link_emb_test).norm(dim=1) * link_logits_test.squeeze().sign()#*(-1)#[:, None].expand_as(link_emb_test)
    interv_link_emb_test = interv_link_emb_test[:, None]

elif args.type_process_emb == 'emb_mlp':
    interv_link_emb_test = link_emb_test
else:
    raise Exception("Use sub or dist!")

auc_test_real = roc_auc_score(link_labels_test.int().to(device), link_logits_test.sigmoid().squeeze().detach().to(device))

feat = interv_link_emb_test.shape[1]
state_dict = torch.load('./best_calib_model/'+ args.model_gnn + '_' + args.dataset + '_in_n_out_.pth')
calib = IN_N_OUT(feat, 5, 128, 0.1,True).to(device)
calib.load_state_dict(state_dict)
calib.eval()
calib = calib.to(device)


output_test = calib(link_logits_test.unsqueeze(1), interv_link_emb_test)
auc_test_pred = roc_auc_score(link_labels_test.int().cpu(), output_test[:,1].sigmoid().squeeze().detach().cpu())
ece_criterion = ECELoss(15).cuda()
ece_test = ece_criterion(output_test, link_labels_test)

metric = ECELoss(15, device).to(device)
ece_test = metric(output_test, link_labels_test)
eces_test = ece_test.item()

aucs_test = auc_test_pred.item()

print(f'ece: {round(eces_test*100, 2)}')
print(f'auc: {round(aucs_test*100, 2)}')
from ogb.linkproppred import Evaluator

evaluator20 = Evaluator(name = 'ogbl-ddi')
evaluator50 = Evaluator(name = 'ogbl-collab')

input_dict = {'y_pred_pos': output_test[:,1].sigmoid().squeeze()[torch.where(link_labels_test == 1)[0]],
                'y_pred_neg': output_test[:,1].sigmoid().squeeze()[torch.where(link_labels_test == 0)[0]]}
print('hits@20: ', evaluator20.eval(input_dict)['hits@20']*100)
print('hits@50: ', evaluator50.eval(input_dict)['hits@50']*100)

# ece: 15.36
# auc:  89.39
# hits@20:  48.96
# hits@50:  66.41