# Import packages
from __future__ import print_function
import torch
import torch.nn as nn

# Import utility functions
# from utils.funcs import complex_mul_taps


class basic_DNN(nn.Module):
    """
    Construct a basic Deep Net class
    """

    def __init__(self, M, num_neurons_encoder, n, num_neurons_decoder, if_bias, if_relu):
        """
        Initialize the basic Deep Net class
        :param M: input/output dimension
        :param num_neurons_encoder: number of neurons in the encoding phase
        :param n: data dimension in the bottleneck layer
        :param num_neurons_decoder: number of neurons in the decoding phase
        :param if_bias: turn on bias mode
        :param if_relu: turn on ReLU mode
        """
        super(basic_DNN, self).__init__()
        # Encoder Module
        self.enc_fc1 = nn.Linear(M, num_neurons_encoder, bias=if_bias)
        self.enc_fc2 = nn.Linear(num_neurons_encoder, n, bias=if_bias)

        # Decoder Module
        self.dec_fc1 = nn.Linear(n, num_neurons_decoder, bias=if_bias)
        self.dec_fc2 = nn.Linear(num_neurons_decoder, M, bias=if_bias)

        if if_relu:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.Tanh()
        self.tanh = nn.Tanh()

    def forward(self, x, noise_dist):
        """
        Define a forward pass
        :param x: input
        :param noise_dist: noise distance
        :return: output
        """
        # Encoding
        x = self.enc_fc1(x)
        x = self.activ(x)
        x = self.enc_fc2(x)

        # Normalize
        x_norm = torch.norm(x, dim=1)
        x_norm = x_norm.unsqueeze(1)
        x = pow(x.shape[1], 0.5) * pow(0.5, 0.5) * x / x_norm

        # Adding Noise
        n = torch.zeros(x.shape[0], x.shape[1])
        for noise_batch_ind in range(x.shape[0]):
            n[noise_batch_ind] = noise_dist.sample()
        n = n.type(torch.FloatTensor)
        x = x + n  # Noise Insertion

        # Decoding
        x = self.dec_fc1(x)
        x = self.activ(x)
        x = self.dec_fc2(x)

        return x


def dnn(**kwargs):
    net = basic_DNN(**kwargs)
    return net
