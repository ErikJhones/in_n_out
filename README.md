# IN-N-OUT
# Preparing enviroment

conda create --name in_n_out python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ # This is a one line command

conda activate in_n_out

pip3 install torch torchvision torchaudio

pip install torch_geometric

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

pip install -U ogb

pip install netcal

pip install matplotlib

# Train GNN model (VGAE)
`python train_gnn_model.py --dataset cora --model_gnn VGAE`

# Train IN-N-OUT
`python train_in_n_out.py --model_gnn VGAE --dataset cora --type_process_emb sub --epochs 20 --lr 0.001`

# Infer IN-N-OUT
`python infer_in_n_out.py --type_process_emb sub --dataset cora --model_gnn VGAE`

# traditional calibration
`python traditional_calibration.py --dataset cora --model_gnn VGAE --type_calibrator hist`

# train and infer IN-N-OUT pubmed
`python train_gnn_model.py --dataset pubmed --model_gnn VGAE`  
ece: 20.41  
auc:  95.56  
hits@20:  31.59  
hits@50:  51.92  

`python traditional_calibration.py --model_gnn VGAE --dataset pubmed --type_calibrator hist`  
ece: 3.15  
auc:  95.45  
hits@20:  0.0  
hits@500:  43.86  

`python train_in_n_out.py --model_gnn VGAE --dataset pubmed --type_process_emb sub --epochs 20 --lr 0.001`  
`python infer_in_n_out.py --model_gnn VGAE --dataset pubmed --type_process_emb sub`  

ece: 2.86  
auc: 95.47  
hits@20:  33.41  
hits@50:  52.61  