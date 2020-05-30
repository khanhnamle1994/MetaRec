# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Encoder(nn.Module):
    """
    Function to define SVAE encoder module
    """
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['rnn_size'], hyper_params['hidden_size']
        )
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    """
    Function to define SVAE decoder module
    """
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(hyper_params['latent_size'], hyper_params['hidden_size'])
        self.linear2 = nn.Linear(hyper_params['hidden_size'], hyper_params['total_items'])
        nn.init.xavier_normal(self.linear1.weight)
        nn.init.xavier_normal(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Model(nn.Module):
    """
    Function to build the SVAE model
    """
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params

        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)

        # Since we don't need padding, our vocabulary size will be "hyper_params['total_items']"
        # and not "hyper_params['total_items'] + 1"
        self.item_embed = nn.Embedding(hyper_params['total_items'], hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.linear1 = nn.Linear(hyper_params['hidden_size'], 2 * hyper_params['latent_size'])
        nn.init.xavier_normal(self.linear1.weight)

        self.tanh = nn.Tanh()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.hyper_params['latent_size']]
        log_sigma = temp_out[:, self.hyper_params['latent_size']:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        """
        Function to do a forward pass
        :param x: the input
        """
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        x = x.view(-1)  # [seq_len]

        x = self.item_embed(x)  # [seq_len x embed_size]
        x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x embed_size]

        rnn_out, _ = self.gru(x)  # [1 x seq_len x rnn_size]
        rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)  # [seq_len x rnn_size]

        enc_out = self.encoder(rnn_out)  # [seq_len x hidden_size]
        sampled_z = self.sample_latent(enc_out)  # [seq_len x latent_size]

        dec_out = self.decoder(sampled_z)  # [seq_len x total_items]
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x total_items]

        return dec_out, self.z_mean, self.z_log_sigma


class VAELoss(torch.nn.Module):
    """
    Function to calculate the VAE loss
    """
    def __init__(self, hyper_params):
        super(VAELoss, self).__init__()
        self.hyper_params = hyper_params

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), -1))

        # Calculate Likelihood
        dec_shape = decoder_output.shape  # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)
        num_ones = float(torch.sum(y_true_s[0, 0]))

        likelihood = torch.sum(
            -1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
            decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        ) / (float(self.hyper_params['batch_size']) * num_ones)

        # Calculate the final VAE loss
        final = (anneal * kld) + likelihood

        return final
