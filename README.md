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
python train_gnn_model.py --dataset cora --model_gnn VGAE

# Train IN-N-OUT
python train_in_n_out.py --model_gnn VGAE --dataset cora --type_process_emb sub --epochs 20 --lr 0.001

# Infer IN-N-OUT
python infer_in_n_out.py --type_process_emb sub --dataset cora --model_gnn VGAE
