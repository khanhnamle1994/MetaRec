import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import data
from data import MovieLens1MDataset, MovieLens20MDataset

import models
from models.afi import AutomaticFeatureInteractionModel
from models.afm import AttentionalFactorizationMachineModel
from models.dcn import DeepCrossNetworkModel
from models.dfm import DeepFactorizationMachineModel
from models.ffm import FieldAwareFactorizationMachineModel
from models.fm import FactorizationMachineModel
from models.fnfm import FieldAwareNeuralFactorizationMachineModel
from models.fnn import FactorizationSupportedNeuralNetworkModel
from models.lr import LogisticRegressionModel
from models.ncf import NeuralCollaborativeFiltering
from models.nfm import NeuralFactorizationMachineModel
from models.pnn import ProductNeuralNetworkModel
from models.wd import WideAndDeepModel
from models.xdfm import ExtremeDeepFactorizationMachineModel

def get_dataset(name, path):
    """
    Choose either MovieLens1M or MovieLens20M datasets
    """

    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    field_dims = dataset.field_dims

    if name == 'lr':
        # Logistic Regression model
        return LogisticRegressionModel(field_dims)

    elif name == 'fm':
        # Factorization Machine model
        return FactorizationMachineModel(field_dims, embed_dim = 16)

    elif name == 'ffm':
        # Field Aware Factorization Machine model
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim = 4)

    elif name == 'fnn':
        # Factorization Supported Neural Network model
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim = 16, mlp_dims = (16, 16), dropout = 0.2)

    elif name == 'wd':
        # Wide and Deep Learning model
        return WideAndDeepModel(field_dims, embed_dim = 16, mlp_dims = (16, 16), dropout = 0.2)

    elif name == 'ipnn':
        # Inner Product Neural Network model
        return ProductNeuralNetworkModel(field_dims, embed_dim = 16, mlp_dims = (16,), method = 'inner', dropout = 0.2)

    elif name == 'opnn':
        # Outer Product Neural Network model
        return ProductNeuralNetworkModel(field_dims, embed_dim = 16, mlp_dims = (16,), method = 'outer', dropout = 0.2)

    elif name == 'dcn':
        # Deep Cross Neural Network model
        return DeepCrossNetworkModel(field_dims, embed_dim = 16, num_layers = 3, mlp_dims = (16, 16), dropout = 0.2)

    elif name == 'nfm':
        # Neural Factorization Machine model
        return NeuralFactorizationMachineModel(field_dims, embed_dim = 64, mlp_dims = (64,), dropouts = (0.2, 0.2))

    elif name == 'ncf':
        # Neural Collaborative Filtering model
        return NeuralCollaborativeFiltering(field_dims, embed_dim = 16, mlp_dims = (16, 16), dropout = 0.2, user_field_idx = dataset.user_field_idx, item_field_idx = dataset.item_field_idx)

    elif name == 'fnfm':
        # Field Aware Neural Factorization Machine model
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim = 4, mlp_dims = (64,), dropouts = (0.2, 0.2))

    elif name == 'dfm':
        # Deep Factorization Machine model
        return DeepFactorizationMachineModel(field_dims, embed_dim = 16, mlp_dims = (16, 16), dropout = 0.2)

    elif name == 'xdfm':
        # Extreme Deep Factorization Machine model
        return ExtremeDeepFactorizationMachineModel(field_dims, embed_dim = 16, cross_layer_sizes = (16, 16), split_half = False, mlp_dims = (16, 16), dropout = 0.2)

    elif name == 'afm':
        # Attentional Factorization Machine model
        return AttentionalFactorizationMachineModel(field_dims, embed_dim = 16, attn_size = 16, dropouts = (0.2, 0.2))

    elif name == 'afi':
        # Automatic Feature Interaction model
        return AutomaticFeatureInteractionModel(field_dims, embed_dim = 32, num_heads = 4, num_layers = 2, mlp_dims = (16, 16), dropouts = (0.2, 0.2))

    else:
        raise ValueError('unknown model name: ' + name)

def train(model, optimizer, data_loader, criterion, device, log_interval = 1000):

    model.train()
    total_loss = 0

    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing = 0, mininterval = 1.0)):

        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):

    model.eval()
    targets, predicts = list(), list()

    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):

            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    return roc_auc_score(targets, predicts)

def main(dataset_name, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)

    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 8)
    valid_data_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = 8)
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 8)

    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)

    auc = test(model, test_data_loader, device)
    print('test auc:', auc)
    torch.save(model, f'{save_dir}/{model_name}.pt')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', help='./ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='afi')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()

    main(args.dataset_name, args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir)
