import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_feat, k=10, c_feat=1.0, c_bias=1.0, writer=None):
        """
        :param n_feat: Feature column
        :param k: Dimensions constant
        :param c_feat: Regularization constant for the features
        :param c_bias: Regularization constant for the biases
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.n_feat = n_feat
        self.c_feat = c_feat
        self.c_bias = c_bias

        # The embedding matrices for the features and the feature's biases
        self.feat = nn.Embedding(n_feat, k)
        self.bias_feat = nn.Embedding(n_feat, 1)

    def __call__(self, train_x):
        """This is the most important function in this script"""
        # Pull out biases
        biases = index_into(self.bias_feat.weight, train_x).squeeze().sum(dim=1)

        # Initialize vector features using the feature weights
        vector_features = index_into(self.feat.weight, train_x)

        # Use factorization machines to pull out the interactions
        interactions = factorization_machine(vector_features).squeeze().sum(dim=1)

        # Final prediction is the sum of biases and interactions
        prediction = biases + interactions
        return prediction

    def loss(self, prediction, target):
        """
        Function to calculate the loss metric
        """
        # Calculate the Mean Squared Error between target and prediction
        loss_mse = F.mse_loss(prediction.squeeze(), target.squeeze())

        # Compute L2 regularization over feature matrices
        prior_feat = l2_regularize(self.feat.weight) * self.c_feat

        # Add the MSE loss and feature regularization to get total loss
        total = (loss_mse + prior_feat)

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


def index_into(arr, idx):
    new_shape = (idx.size()[0], idx.size()[1], arr.size()[1])
    return arr[idx.resize(torch.numel(idx.data))].view(new_shape)


def factorization_machine(v, x=None):
    """
    Takes an input 2D matrix v of n vectors, each d-dimensional
    :param v: (batchsize, n_features, dim)
    :param x: (batchsize, n_features) functions as a weight array, assumed to be 1 if missing
    :return: output that is d-dimensional
    """
    batchsize = v.size()[0]
    n_features = v.size()[1]
    n_dim = v.size()[2]

    if x is None:
        x = Variable(torch.ones(v.size()))
    else:
        x = x.expand(batchsize, n_features, n_dim)

    # Uses Rendle's trick for computing pairs of features in linear time
    t0 = (v * x).sum(dim=1) ** 2.0
    t1 = (v ** 2.0 * x ** 2.0).sum(dim=1)
    return 0.5 * (t0 - t1)
