import torch
from torch import nn
import torch.nn.functional as F


class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, k=10, c_bias=1.0, c_kld=1.0, writer=None):
        """
        :param n_user: User column
        :param n_item: Item column
        :param k: Dimensions constant
        :param c_bias: Regularization constant for the biases
        :param c_kld: Regularization constant for KL divergence
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_bias = c_bias
        self.c_kld = c_kld

        # We've added new terms here: Embedding matrices for the Mu and Log Variance values of users and items
        self.user_mu = nn.Embedding(n_user, k)
        self.user_lv = nn.Embedding(n_user, k)
        self.item_mu = nn.Embedding(n_item, k)
        self.item_lv = nn.Embedding(n_item, k)

        # The embedding matrices for the user's biases and the item's biases
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)

        # Initialize the bias tensors
        self.bias = nn.Parameter(torch.ones(1))

    def __call__(self, train_x):
        """This is the most important function in this script"""
        # These are the user and item indices
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]

        # *NEW: Stochastically-sampled user & item vectors
        vector_user = sample_gaussian(self.user_mu(user_id), self.user_lv(user_id))
        vector_item = sample_gaussian(self.item_mu(item_id), self.item_lv(item_id))

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # The user-item interaction is a dot product between the user and item vectors
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)

        # Final prediction is the sum of the user-item interaction with the biases
        prediction = ui_interaction + biases
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

        # *NEW: Compute the KL-Divergence loss over the Mu and Log-Variance for user and item matrices
        user_kld = gaussian_kldiv(self.user_mu.weight, self.user_lv.weight) * self.c_kld
        item_kld = gaussian_kldiv(self.item_mu.weight, self.item_lv.weight) * self.c_kld

        # Add up the MSE loss + user & item biases regularization + user & item KL-Divergence loss
        total = loss_mse + prior_bias_user + prior_bias_item + user_kld + item_kld

        # This logs all local variables to tensorboard
        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        return total


def l2_regularize(array):
    """
    Function to compute L2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss


def sample_gaussian(mu, log_var):
    """
    Function to compute a Gaussian Sample Distribution
    """
    var = log_var.mul(0.5).exp_()
    eps = torch.FloatTensor(var.size()).normal_()
    return mu + eps * var


def gaussian_kldiv(mu, log_var):
    """
    Function to compute a Gaussian KL Divergence loss
    """
    kld = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    kldloss = torch.sum(kld).mul_(-0.5)
    return kldloss
