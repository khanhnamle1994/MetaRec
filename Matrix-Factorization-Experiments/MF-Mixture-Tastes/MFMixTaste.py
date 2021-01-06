import torch
from torch import nn
import torch.nn.functional as F


class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, k=10, c=4, c_vector=1.0, c_bias=1.0, writer=None):
        """
        :param n_user: User column
        :param n_item: Item column
        :param k: Dimensions constant
        :param c: Dimension of taste/attention vectors
        :param c_vector: Regularization constant
        :param c_bias: Regularization constant for the biases
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.c = c
        self.n_user = n_user
        self.n_item = n_item
        self.c_bias = c_bias
        self.c_vector = c_vector

        # Embedding matrices for the user's biases and the item's biases
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)

        # Initialize the bias tensors
        self.bias = nn.Parameter(torch.ones(1))

        # The embedding matrix for item is learned and fit by PyTorch
        self.item = nn.Embedding(n_item, k)

        # **NEW: Initialize user taste & attention vectors
        user_taste = torch.zeros(n_user, k, c)
        user_attention = torch.zeros(n_user, k, c)
        user_taste.data.normal_(0, 1.0 / n_user)
        user_attention.data.normal_(0, 1.0 / n_user)

        # The embedding matrices for taste and attention are learned and fit by PyTorch
        self.user_taste = nn.Parameter(user_taste)
        self.user_attention = nn.Parameter(user_attention)

    def __call__(self, train_x):
        """This is the most important function in this script"""
        # These are the user and item indices
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]

        # Initialize a vector item using the item indices
        vector_item = self.item(item_id)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # **NEW: Initialize the user taste & attention matrices using the user IDs
        user_taste = self.user_taste[user_id]
        user_attention = self.user_attention[user_id]

        vector_itemx = vector_item.unsqueeze(2).expand_as(user_attention)
        attention = F.softmax(user_attention * vector_itemx, dim=1)
        attentionx = attention.sum(2).unsqueeze(2).expand_as(user_attention)

        # Calculate the weighted preference to be the dot product of the user taste and attention
        weighted_preference = (user_taste * attentionx).sum(2)
        # This is a dot product of the weighted preference and vector item
        dot = (weighted_preference * vector_item).sum(1)

        # Final prediction is the sum of the biases and the dot product above
        prediction = dot + biases
        return prediction

    def loss(self, prediction, target):
        """
        Function to calculate the loss metric
        """
        # Calculate the Mean Squared Error between target and prediction
        loss_mse = F.mse_loss(prediction.squeeze(), target.squeeze())

        # Compute L2 regularization over the biases for user and the biases for item matrices
        prior_bias_user = l2_regularize(self.bias_user.weight) * self.c_bias
        prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias

        # Compute L2 regularization over the user tastes and user attentions matrix
        prior_taste = l2_regularize(self.user_taste) * self.c_vector
        prior_attention = l2_regularize(self.user_attention) * self.c_vector

        # Compute L2 regularization over item matrix
        prior_item = l2_regularize(self.item.weight) * self.c_vector

        # Add up the MSE loss + user & item biases regularization + item regularization + user taste & attention
        # regularization
        total = (loss_mse + prior_bias_item + prior_bias_user + prior_taste + prior_attention + prior_item)

        # This logs all local variables to tensorboard
        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        return total


def l2_regularize(array):
    """
    Function to do L2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss
