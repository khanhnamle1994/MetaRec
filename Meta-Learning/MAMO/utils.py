# Import packages
import numpy as np
import os
import torch
import pickle
import random
from math import log2
from torch.utils.data import Dataset
from copy import deepcopy

# Configuration Settings
config = {'n_rate': 6, 'n_year': 81, 'n_genre': 25, 'n_director': 2186,
          'n_gender': 2, 'n_age': 7, 'n_occupation': 21, 'sample_size': 20}

default_info = {
    'movielens': {'n_y': 5, 'u_in_dim': 3, 'i_in_dim': 4}
}


def to_torch(in_list):
    """
    Convert Numpy array to PyTorch tensor
    """
    return torch.from_numpy(np.array(in_list))


def train_test_user_list(rand=True, random_state=32, train_test_split_ratio=0.8, store=False):
    """
    Split the data into train and test sets
    :param rand: Boolean value to turn on randomization mode
    :param random_state:
    :param train_test_split_ratio: Percentage of train set
    :param store: Boolean value to turn on storage mode
    :return: Train and test data
    """
    path = 'data_prep/processed_data/raw/'
    path_store = 'data_prep/'
    len_user = int(len(os.listdir(path)) / 4)
    training_size = int(len_user * train_test_split_ratio)
    user_id_list = list(range(1, len_user))

    # Random shuffling
    if rand:
        random.shuffle(user_id_list)
    else:
        random.seed(random_state)
        random.shuffle(user_id_list)

    train_user_set, test_user_set = user_id_list[:training_size], user_id_list[training_size:]

    # Store the train and test user data
    if store:
        pickle.dump(train_user_set, open(path_store + 'train_user_set.p', 'wb'))
        pickle.dump(test_user_set, open(path_store + 'test_user_set.p', 'wb'))

    return train_user_set, test_user_set


def load_user_info(user_id, support_size=16, query_size=4, device=torch.device('cpu')):
    """
    Load user information
    :param user_id: User ID
    :param support_size: Size of the support set
    :param query_size: Size of the query set
    :param device: Choice of device
    :return: Support and query sets containing user info
    """
    path = 'data_prep/processed_data/raw/'

    # Load Pickle files that include user info, item info, ratings, and cold labels
    u_x1 = pickle.load(open('{}sample_{}_x1.p'.format(path, str(user_id)), 'rb'))
    u_x2 = pickle.load(open('{}sample_{}_x2.p'.format(path, str(user_id)), 'rb'))
    u_y = pickle.load(open('{}sample_{}_y.p'.format(path, str(user_id)), 'rb'))
    u_y0 = pickle.load(open('{}sample_{}_y0.p'.format(path, str(user_id)), 'rb'))

    u_x1 = np.tile(u_x1, (config['sample_size'], 1))
    u_y = u_y - 1

    # Generate support set and query set
    sup_x1, que_x1 = to_torch(u_x1[:support_size]).to(device), \
                     to_torch(u_x1[support_size:support_size + query_size]).to(device)
    sup_x2, que_x2 = to_torch(u_x2[:support_size]).to(device), \
                     to_torch(u_x2[support_size:support_size + query_size]).to(device)
    sup_y, que_y = to_torch(u_y[:support_size]).to(device), \
                   to_torch(u_y[support_size:support_size + query_size]).to(device)
    sup_y0, que_y0 = to_torch(u_y0[:support_size]).to(device), \
                     to_torch(u_y0[support_size:support_size + query_size]).to(device)

    return sup_x1, sup_x2, sup_y, sup_y0, que_x1, que_x2, que_y, que_y0


class UserDataLoader(Dataset):
    """
    Create a class to load user data
    """

    def __init__(self, x1, x2, y, y0, transform=None):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.y0 = y0
        self.transform = transform

    def __len(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        user_info = self.x1[idx]
        item_info = self.x2[idx]
        ratings = self.y[idx]
        cold_labels = self.y0[idx]

        return user_info, item_info, ratings, cold_labels


def get_params(param_list):
    """
    Collect model parameters
    :param param_list: parameter list
    :return: model parameters
    """
    params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.data)
            params.append(value)
            del value
        count += 1
    return params


def get_zeros_like_params(param_list):
    zeros_like_params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(torch.zeros_like(param.data))
            zeros_like_params.append(value)
        count += 1
    return zeros_like_params


def init_params(param_list, init_values):
    """
    Initialize model parameters
    :param param_list: parameter list
    :param init_values: initialization values list
    :return:
    """
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count])
            init_count += 1
        count += 1


def init_u_mem_params(param_list, init_values, bias_term, tao):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values)
            init_count += 1
        count = 1


def init_ui_mem_params(param_list, init_values):
    count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values)
        count += 1


def get_grad(param_list):
    """
    Collect model gradients
    :param param_list: parameter list
    :return: model gradients
    """
    count = 0
    param_grads = []
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.grad)
            param_grads.append(value)
            del value
        count += 1
    return param_grads


def grads_sum(raw_grads_list, new_grads_list):
    return [raw_grads_list[i] + new_grads_list[i] for i in range(len(raw_grads_list))]


def update_parameters(params, grads, lr):
    return [params[i] - lr * grads[i] for i in range(len(params))]


def activation_func(name):
    """
    Choice of activation function
    :param name: name of activation function
    """
    name = name.lower()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "softmax":
        return torch.nn.Softmax()
    elif name == "leaky_relu":
        return torch.nn.LeakyReLU(0.1)
    else:
        return torch.nn.Sequential()


def mae(ground_truth, test_result):
    """
    Calculate Mean Absolute Error
    :param ground_truth: Ground-truth rating
    :param test_result: Predicted rating
    """
    if len(ground_truth) > 0:
        pred_y = torch.argmax(test_result, dim=1)
        sub = ground_truth - pred_y
        abs_sub = torch.abs(sub)
        out = torch.mean(abs_sub.float(), dim=0)
    else:
        out = 1
    return out


def ndcg(ground_truth, test_result, top_k=10):
    """
    Calculate Normalized Discounted Cumulative Gain
    :param ground_truth: Ground-truth rating
    :param test_result: Predicted rating
    :param top_k: Number of items to be ranked
    """
    pred_y = torch.argmax(test_result, dim=1)
    sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
    sort_pred_y, sort_pred_y_index = pred_y.clone().detach().sort(descending=True)
    pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
    top_pred_y, _ = pred_sort_y.sort(descending=True)

    ideal_dcg = 0
    n = 1

    for value in sort_real_y[:top_k]:
        idcg = (2 ** float(value + 1) - 1) / log2(n + 1)
        ideal_dcg += idcg
        n += 1

    pred_dcg = 0
    n = 1

    for value in top_pred_y:
        pdcg = (2 ** float(value + 1) - 1) / log2(n + 1)
        pred_dcg += pdcg
        n += 1

    ndcg = pred_dcg / ideal_dcg
    return ndcg
