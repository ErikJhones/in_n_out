import torch
import torch.optim as optim
from torch.optim import Adam, AdamW

from calibrator import IN_N_OUT
from utils_calib import *
from utils import accuracy
from models import *

from tqdm import tqdm

import os
import numpy as np
import time
import argparse
from args_calib import get_args
from datasets import *

# Arguments
args = get_args()

device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

seed = 42
torch.manual_seed(42)
np.random.seed(42)

data, _ = choose_dataset(args.dataset)
data = data.to(device)

path = './models/'

model = choose_model(data, args.model_gnn).to(device)
    
ckpt_destilado = torch.load(path + args.model_gnn + '_'+args.dataset+ '_seed_' + str(seed) + '.pth.tar', map_location=torch.device(device))
model.load_state_dict(ckpt_destilado["model_state"]) 

for para in model.parameters():
    para.requires_grad = False


edge_index_val, link_logits_val, link_labels_val, n_val = get_model_data(model, data, 'val', device)
edge_index_test, link_logits_test, link_labels_test, n_test = get_model_data(model, data, 'test', device)
edge_index_train, link_logits_train, link_labels_train, n_train = get_model_data(model, data, 'train', device)

node_embeddings = model.embedding(data.x, data.train_pos_edge_index)

link_emb_val = agr_emb(edge_index_val, node_embeddings)
link_emb_test = agr_emb(edge_index_test, node_embeddings)
link_emb_train= agr_emb(edge_index_train, node_embeddings)

print(f'val size: {edge_index_val.shape}')
print(f'test size: {edge_index_test.shape}')
print(f'train size: {edge_index_train.shape}')

from pathlib import Path

my_file = Path("./inter_emb_model/intervential_emb_train_"+args.dataset+"_"+args.model_gnn+".pt")
if my_file.is_file():
    intervential_emb_val = torch.load('./inter_emb_model/intervential_emb_val_'+args.dataset+'_'+args.model_gnn+'.pt')
    intervential_emb_test = torch.load('./inter_emb_model/intervential_emb_test_'+args.dataset+'_'+args.model_gnn+'.pt')
    intervential_emb_train = torch.load('./inter_emb_model/intervential_emb_train_'+args.dataset+'_'+args.model_gnn+'.pt')
else:

    intervential_emb_val = []
    for k in tqdm(get_interventional_emb(data.train_pos_edge_index, edge_index_val, model, data.x, device, 'val')):
        intervential_emb_val.append(list(k.detach().numpy()))
    intervential_emb_val = torch.tensor(intervential_emb_val)

    intervential_emb_test = []
    for k in tqdm(get_interventional_emb(data.train_pos_edge_index, edge_index_test, model, data.x, device, 'test')):
        intervential_emb_test.append(list(k.detach().numpy()))
    intervential_emb_test = torch.tensor(intervential_emb_test)

    intervential_emb_train = []
    for k in tqdm(get_interventional_emb(data.train_pos_edge_index, edge_index_train, model, data.x, device, 'train')):
        intervential_emb_train.append(list(k.detach().numpy()))
    intervential_emb_train = torch.tensor(intervential_emb_train)

    torch.save(intervential_emb_val, './inter_emb_model/intervential_emb_val_'+args.dataset+'_'+args.model_gnn+'.pt')
    torch.save(intervential_emb_test, './inter_emb_model/intervential_emb_test_'+args.dataset+'_'+args.model_gnn+'.pt')
    torch.save(intervential_emb_train, './inter_emb_model/intervential_emb_train_'+args.dataset+'_'+args.model_gnn+'.pt')

if args.type_process_emb == 'sub':
    interv_link_emb_val = (intervential_emb_val - link_emb_val)
    interv_link_emb_test = (intervential_emb_test - link_emb_test)
    interv_link_emb_train = (intervential_emb_train - link_emb_train)

elif args.type_process_emb == 'dist':
    interv_link_emb_val = torch.pairwise_distance(intervential_emb_val, link_emb_val, p=1).unsqueeze(1)
    interv_link_emb_test = torch.pairwise_distance(intervential_emb_test, link_emb_test, p=1).unsqueeze(1)
    interv_link_emb_train = torch.pairwise_distance(intervential_emb_train, link_emb_train, p=2).unsqueeze(1)

elif args.type_process_emb == 'sub_signed':
    interv_link_emb_val = (intervential_emb_val - link_emb_val) * link_logits_val.sign()#.unsqueeze(1)
    interv_link_emb_test = (intervential_emb_test - link_emb_test) * link_logits_test.sign()#.unsqueeze(1)
    interv_link_emb_train = (intervential_emb_train - link_emb_train) * link_logits_train.sign()#.unsqueeze(1)

elif args.type_process_emb == 'dist_signed':
    interv_link_emb_val = (intervential_emb_val - link_emb_val).norm(dim=1) * link_logits_val.squeeze().sign()#*(-1)#[:, None].expand_as(link_emb_val)
    interv_link_emb_val = interv_link_emb_val[:, None]

    interv_link_emb_test = (intervential_emb_test - link_emb_test).norm(dim=1) * link_logits_test.squeeze().sign()#*(-1)#[:, None].expand_as(link_emb_test)
    interv_link_emb_test = interv_link_emb_test[:, None]

    interv_link_emb_train = (intervential_emb_train - link_emb_train).norm(dim=1) * link_logits_train.squeeze().sign()#*(-1)#[:, None].expand_as(link_emb_test)
    interv_link_emb_train = interv_link_emb_train[:, None]
elif args.type_process_emb == 'emb_mlp':
    interv_link_emb_val = link_emb_val
    interv_link_emb_test = link_emb_test
    interv_link_emb_train = link_emb_train
else:
    raise Exception("Use sub or dist!")


criterion = torch.nn.CrossEntropyLoss()

def train(logits_train, logits_val, logits_test, our_calibrator, optimizer, emb_train, emb_val, emb_test, labels_train, labels_val, labels_test,  reg, sign=False):
    t = time.time()
    our_calibrator.train()
    optimizer.zero_grad()

    output_train = our_calibrator(logits_train, emb_train)
    ece_criterion = ECELoss(15, device).to(device)

    output_train, labels_train = output_train.to(device), labels_train.to(device)
    labels_val = labels_val.to(device)
    labels_test = labels_test.to(device)

    ece = ece_criterion(output_train.to(device), labels_train.to(device))
    if sign:
        loss_train = criterion(output_train.to(device), labels_train.to(device)) + (reg[1].to(device) * ece)
    else:
        loss_train = criterion(output_train, labels_train) + (reg[0] * intra_distance_loss(output_train, labels_train)) + (reg[1] *ece)
    acc_train = accuracy(output_train, labels_train)

    loss_train.backward()
    optimizer.step()
    
    with torch.no_grad():
        our_calibrator.eval()
        output_val = our_calibrator(logits_val, emb_val)
        output_test = our_calibrator(logits_test, emb_test)
        loss_val = criterion(output_val, labels_val)
        loss_test = criterion(output_test, labels_test)
        acc_val = accuracy(output_val, labels_val)
        acc_test = accuracy(output_test, labels_test.detach())
        ece_test = ece_criterion(output_test, labels_test)
        ece_val = ece_criterion(output_val, labels_val)
        auc_test = roc_auc_score(link_labels_test.int().to('cpu'), output_test[:,1].sigmoid().squeeze().to('cpu'))
        auc_val = roc_auc_score(link_labels_val.int().to('cpu'), output_val[:,1].sigmoid().squeeze().to('cpu'))
        auc_train = roc_auc_score(link_labels_train.int().to('cpu'), output_train[:,1].sigmoid().squeeze().to('cpu'))

    
    return loss_train, loss_val, loss_test, auc_train, auc_val, auc_test, ece, ece_val, ece_test

torch.manual_seed(args.seed)
feat = interv_link_emb_train.shape[1]
calib = IN_N_OUT(feat, 5, 128, 0.1,True).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, calib.parameters()),
                        lr=args.lr, weight_decay=args.weight_decay)
best_val = 100
best_epoch = 0
for epoch in range(args.epochs):
    loss_train, loss_val, loss_test, auc_train, auc_val, auc_test, ece, ece_val, ece_test = train(link_logits_train.unsqueeze(1).detach(),
                                                    link_logits_val.unsqueeze(1).detach(),
                                                    link_logits_test.unsqueeze(1).detach(),
                                                    calib,
                                                    optimizer, 
                                                    interv_link_emb_train.detach(),
                                                    interv_link_emb_val.detach(),
                                                    interv_link_emb_test.detach(),
                                                    link_labels_train.detach(),
                                                    link_labels_val.detach(),
                                                    link_labels_test.detach(),
                                                    torch.tensor((1.0, 1.0)).to(device),
                                                    True)
      
    print(f'epoch: {epoch}',
        f'loss_train: {loss_train.item():.4f}',
        f'auc_train: {auc_train:.4f}',
        f'ece_train: {ece.item():.4f}',
        f'loss_val: {loss_val.item():.4f}',
        f'auc_val: {auc_val:.4f}',
        f'ece_val: {ece_val.item():.4f}',
        f'loss_test: {loss_test.item():.4f}',
        f'auc_test: {auc_test:.4f}',
        f'ece_test: {ece_test.item():.4f}',)
    
    if ece_val < best_val:
        best_val = ece_val
        print(epoch, best_val)
        torch.save(calib.state_dict(), './best_calib_model/'+args.model_gnn+ '_' + args.dataset + '_in_n_out_.pth')
