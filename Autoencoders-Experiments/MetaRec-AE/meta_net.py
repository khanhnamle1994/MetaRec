# Import packages
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import functional as F


class meta_Net_DNN(nn.Module):
    """
    Construct a Meta Deep Network class
    """

    def __init__(self, if_relu):
        """
        Initialize a Meta Deep Network class that only gets params from other nets' params
        :param if_relu: turn on ReLU mode
        """
        super(meta_Net_DNN, self).__init__()
        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, var, if_bias, noise_dist):
        """
        Define a forward pass
        :param x: input
        :param var: list of variables
        :param if_bias: turn on bias mode
        :param noise_dist: noise distance
        :return: output
        """
        idx_int = 0

        if if_bias: gap = 2
        else: gap = 1

        idx = idx_int

        while idx < len(var):
            if idx > idx_int:  # no activation from the beginning
                if idx == gap * 2 + idx_int:  # after the last layer of the encoder
                    pass
                else:
                    x = self.activ(x)

            if idx == idx_int:
                if if_bias:
                    w1, b1 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w1, b1)
                    idx += 2
                else:
                    w1 = var[idx]  # weight
                    x = F.linear(x, w1)
                    idx += 1

            elif idx == gap * 1 + idx_int:
                if if_bias:
                    w2, b2 = var[idx], var[idx + 1]  # weight and bias
                    x = F.linear(x, w2, b2)
                    idx += 2
                else:
                    w2 = var[idx]  # weight
                    x = F.linear(x, w2)
                    idx += 1

            elif idx == gap * 2 + idx_int:
                # Normalize the data
                x_norm = torch.norm(x, dim=1)
                x_norm = x_norm.unsqueeze(1)
                x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm

                # Add noise
                n = torch.zeros(x.shape[0], x.shape[1])
                for noise_batch_ind in range(x.shape[0]):
                    n[noise_batch_ind] = noise_dist.sample()
                n = n.type(torch.FloatTensor)
                x = x + n  # Noise Insertion

        return x


def meta_dnn(**kwargs):
    net = meta_Net_DNN(**kwargs)
    return net
