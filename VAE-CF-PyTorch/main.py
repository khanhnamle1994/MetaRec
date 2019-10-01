import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models
import data
import metric

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='ml-20m',
                    help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
