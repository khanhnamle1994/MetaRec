import torch
from torch import nn
import torch.nn.functional as F

class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, k=18, c=4, c_vector=1.0, c_bias=1.0, writer=None):
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

        # These are the bias vectors
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)
        self.bias = nn.Parameter(torch.ones(1))

        # This is learned and fit by PyTorch
        self.item = nn.Embedding(n_item, k)

        # **NEW: user taste & attention vectors
        user_taste = torch.zeros(n_user, k, c)
        user_attnd = torch.zeros(n_user, k, c)
        user_taste.data.normal_(0, 1.0 / n_user)
        user_attnd.data.normal_(0, 1.0 / n_user)

        self.user_taste = nn.Parameter(user_taste)
        self.user_attnd = nn.Parameter(user_attnd)

    def __call__(self, train_x):
        # These are the user and item indices
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]

        # vector item
        vector_item = self.item(item_id)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # **NEW: user taste & attention
        user_taste = self.user_taste[user_id]
        user_attnd = self.user_attnd[user_id]
        vector_itemx = vector_item.unsqueeze(2).expand_as(user_attnd)
        attention = F.softmax(user_attnd * vector_itemx, dim=1)
        attentionx = attention.sum(2).unsqueeze(2).expand_as(user_attnd)
        weighted_preference = (user_taste * attentionx).sum(2)
        # This is a dot product of the weighted preference and vector item
        dot = (weighted_preference * vector_item).sum(1)

        # Add bias prediction and the dot product above
        prediction = dot + biases
        return prediction

    def loss(self, prediction, target):
        '''
        Function to calculate the loss metric
        '''
        # MSE error between target = R_ui and prediction = p_u * q_i
        loss_mse = F.mse_loss(prediction.squeeze(), target.squeeze())

        # Add new regularization to the biases
        prior_bias_user =  l2_regularize(self.bias_user.weight) * self.c_bias
        prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias

        # Add new regularzation to the user tastes and user attentions
        prior_taste =  l2_regularize(self.user_taste) * self.c_vector
        prior_attnd =  l2_regularize(self.user_attnd) * self.c_vector

        # Compute L2 reularization over item (Q) matrix
        prior_item = l2_regularize(self.item.weight) * self.c_vector

        # Add up the MSE loss + user & item biases regularization + item regularzation + user taste & attention regularzation
        total = (loss_mse + prior_bias_item + prior_bias_user + prior_taste + prior_attnd + prior_item)

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
