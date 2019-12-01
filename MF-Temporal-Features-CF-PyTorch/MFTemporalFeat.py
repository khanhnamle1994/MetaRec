import torch
from torch import nn
import torch.nn.functional as F

class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, n_occu, n_rank, k=18, kt=2, c_temp=1.0, c_vector=1.0, c_bias=1.0, c_ut=1.0, writer=None):
        '''
        Function to initialize the MF class
        '''
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are simple hyperparameters
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_bias = c_bias
        self.c_vector = c_vector
        self.c_temp = c_temp
        self.c_ut = c_ut

        # These are learned and fit by PyTorch
        self.user = nn.Embedding(n_user, k)
        self.item = nn.Embedding(n_item, k)

        # These are the bias vectors
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)
        self.bias = nn.Parameter(torch.ones(1))

        # Occupation vectors
        self.occu = nn.Embedding(n_occu, k)

        # **NEW: temporal vectors
        self.temp = nn.Embedding(n_rank, k)
        self.user_temp = nn.Embedding(n_user, kt)
        self.temp = nn.Embedding(n_rank, kt)

    def __call__(self, train_x):
        '''This is the most important function in this script'''
        # These are the user indices, and correspond to "u" variable
        user_id = train_x[:, 0]
        # These are the item indices, correspond to the "i" variable
        item_id = train_x[:, 1]
        # These are the occupation indicies, correspond to the "o" variable
        occu_id = train_x[:, 3]

        # Vector user
        vector_user = self.user(user_id)
        # Vector item
        vector_item = self.item(item_id)
        # Vector occupation
        vector_occu = self.occu(occu_id)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # This is a dot product to calculate the user-item interaction
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)
        # This is a dot product to calculate the user-occupation interaction
        uo_interaction = torch.sum(vector_user * vector_occu, dim=1)

        # **NEW: user-time interaction
        rank = train_x[:, 2]
        vector_user_temp = self.user_temp(user_id)
        vector_temp = self.temp(rank)
        # This is a dot product to calculate the user-time interaction
        ut_interaction = torch.sum(vector_user_temp * vector_temp, dim=1)

        # Final prediction is the sum of all these interactions with the biases
        prediction = ui_interaction + uo_interaction + ut_interaction + biases
        return prediction

    def loss(self, prediction, target):
        '''
        Function to calculate the loss metric
        '''
        # MSE error between target = R_ui and prediction = p_u * q_i
        loss_mse = F.mse_loss(prediction.squeeze(), target.squeeze())

        # Add regularization to the biases
        prior_bias_user =  l2_regularize(self.bias_user.weight) * self.c_bias
        prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias

        # Compute L2 reularization over user, item, and occupation matrices
        prior_user =  l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector
        prior_occu = l2_regularize(self.occu.weight) * self.c_vector

        # New: total variation regularization
        prior_ut = l2_regularize(self.user_temp.weight) * self.c_ut
        prior_tv = total_variation(self.temp.weight) * self.c_temp

        # Add up the MSE loss + user & item regularization + user & item biases regularization + occupation regularzation + total variation regularzation
        total = loss_mse + prior_user + prior_item + prior_ut +  prior_bias_item + prior_bias_user +  prior_occu + prior_tv

        # This logs all local variables to tensorboard
        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        return total

def l2_regularize(array):
    '''
    Function to do L2 regularization
    '''
    loss = torch.sum(array ** 2.0)
    return loss

def total_variation(array):
    '''
    Function to calculate total variation
    '''
    return torch.sum(torch.abs(array[:, :-1] - array[:, 1:]))
