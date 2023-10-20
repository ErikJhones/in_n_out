import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-8,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model_gnn', type=str, default="VGAE",
                        choices=["VGAE", "PEG", "GraphSAGE"],
                        help='model to use.')
    parser.add_argument('--type_process_emb', type=str, default="sub",
                        choices=['sub', 'dist', 'none'],
                        help='feature-type')
                        
    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
