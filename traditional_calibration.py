from netcal.binning.BBQ import BBQ
from netcal.binning.HistogramBinning import HistogramBinning
from netcal.binning.IsotonicRegression import IsotonicRegression
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE
import torch
from datasets import *
from models import *
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
import numpy as np


def traditional_calibrator(link_logits, link_labels,  link_logits_test, tipo='iso'):
    if tipo=='iso':
        lr = IsotonicRegression()
        lr.fit(link_logits, link_labels)
        lr_test_predictions = lr.transform(link_logits_test)
    elif tipo=='temp':
        lr = TemperatureScaling()
        lr.fit(link_logits, link_labels)
        lr_test_predictions = lr.transform(link_logits_test)
    elif tipo=='bbq':
        lr = BBQ()
        lr.fit(link_logits, link_labels)
        lr_test_predictions = lr.transform(link_logits_test)
    elif tipo=='hist':
        lr = HistogramBinning(bins=16, equal_intervals=True)
        lr.fit(link_logits, link_labels)
        lr_test_predictions = lr.transform(link_logits_test)
    
    return lr_test_predictions

device_string = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

seed = 42
dataset_name = "cora"
gnn = 'VGAE'
type_calibrator = 'hist'
path="models/"

data, _ = choose_dataset(dataset_name)
data = data.to(device)


model = choose_model(data, 'VGAE').to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_destilado = torch.load(path + gnn + '_'+dataset_name+ '_seed_' + str(seed) + '.pth.tar', map_location=torch.device(device))
model.load_state_dict(ckpt_destilado["model_state"]) 

for para in model.parameters():
    para.requires_grad = False

edge_index_val, link_logits_val, link_labels_val, n_val = get_model_data(model, data, 'val', device)
edge_index_test, link_logits_test, link_labels_test, n_test = get_model_data(model, data, 'test', device)
edge_index_train, link_logits_train, link_labels_train, n_train = get_model_data(model, data, 'train', device)

eces_test = []
aucs_test = []

link_confidences_val = torch.cat([1-link_logits_val.unsqueeze(1).sigmoid(), link_logits_val.unsqueeze(1).sigmoid()], dim=1)
link_confidences_test = torch.cat([1-link_logits_test.unsqueeze(1).sigmoid(), link_logits_test.unsqueeze(1).sigmoid()], dim=1)
link_confidences_train = torch.cat([1-link_logits_train.unsqueeze(1).sigmoid(), link_logits_train.unsqueeze(1).sigmoid()], dim=1)


output_test = torch.tensor(traditional_calibrator(link_confidences_train.cpu().numpy(),
       link_labels_train.cpu().numpy(),
       link_confidences_test.cpu().numpy(),
        tipo=type_calibrator)).unsqueeze(1)

metric = ECE(15)
ece = metric.measure(torch.cat([1-output_test, output_test], dim=1).numpy(), link_labels_test.numpy())
print(f'ece: {round(ece*100,2)}')

print('auc: ', roc_auc_score(link_labels_test.int().cpu(), output_test.squeeze().detach().cpu())*100)
from ogb.linkproppred import Evaluator

evaluator20 = Evaluator(name = 'ogbl-ddi')
evaluator50 = Evaluator(name = 'ogbl-collab')

input_dict = {'y_pred_pos': output_test.squeeze()[torch.where(link_labels_test == 1)[0]],
                'y_pred_neg': output_test.squeeze()[torch.where(link_labels_test == 0)[0]]}
print('hits@20: ', evaluator20.eval(input_dict)['hits@20']*100)
print('hits@500: ', evaluator50.eval(input_dict)['hits@50']*100)
