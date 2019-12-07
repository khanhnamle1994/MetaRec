import torch
from torch import nn
import torch.nn.functional as F

# this function multiplies a sparse matrix with a dense matrix
from torch_sparse import spmm
from torch.nn import Parameter

class MFAE(nn.Module):
    itr = 0
    frac = 0.5

    def __init__(self, n_encoder_item, n_decoder_item, k=18, c_vector=1.0, writer=None):
        '''
        Function to initialize the MFAE class
        '''
        super(MFAE, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are simple hyperparameters
        self.k = k
        self.c_vector = c_vector
        self.n_item = n_item
        self.n_encoder_item = n_encoder_item

        # These are encoder vectors
        self.encoder_bias = Parameter(torch.randn(n_encoder_item, 1) * 1e-6)
        self.encoder_vect = Parameter(torch.randn(n_encoder_item, k) * 1e-6)

        # These are decder vectors
        self.decoder_bias = Parameter(torch.randn(n_decoder_item, 1) * 1e-6)
        self.decoder_vect = Parameter(torch.randn(n_decoder_item, k) * 1e-6)

    def __call__(self, indices):
        # first column is user index, 2nd is item index and 3rd is an index over (item, rating_type), and not just item.
        # In the encoder, we use user index and item-rating-type index
        idx = torch.transpose(indices[:, [0, 4]], 1, 0)
        n_user_max = indices[:, 0].max() + 1

        # The encoder does a matrix multiply between a 0 or 1 flag if a feature is present for a user and the dense item representation matrix
        values = torch.ones(len(indices))
        count = 1 + torch.bincount(indices[:, 0], minlength=n_user_max).float()

        # this mask forces us to stochastically use half the player's ratings to predict them all
        mask = torch.rand(len(indices)) > self.frac
        user_bias_sum = spmm(idx[:, mask], values[mask], n_user_max, self.encoder_bias)
        user_vect_sum = spmm(idx[:, mask], values[mask], n_user_max, self.encoder_vect)
        user_bias_mean = user_bias_sum / count[:, None]
        user_vect_mean = user_vect_sum / count[:, None]
        # Note user_vector is of size (max(user_idx in batch), k) and not (batchsize, k)!

        # Now we're in the decoder. These are the user and item indices
        user_idx = indices[:, 0]
        item_idx = indices[:, 1]

        # Extract user/item bias/vectors
        # Note: we're using a different item representation in the decoder than the encoder
        user_bias = user_bias_mean[user_idx]
        user_vect = user_vect_mean[user_idx]
        item_bias = self.decoder_bias[item_idx]
        item_vect = self.decoder_vect[item_idx]

        # Compute likelihood
        user_item = (item_vect * user_vect).sum(dim=1)[:, None]
        log_odds = user_bias + item_bias + user_item
        return log_odds

    def loss(self, log_odds, target):
        '''
        Function to calculate the loss metric
        '''
        # MSE error between target and log odds
        loss_mse = F.mse_loss(log_odds, target.float())

        # Compute regularization for the encoder and the decoder
        prior_ie = l2_regularize(self.encoder_vect) * self.c_vector
        prior_id = l2_regularize(self.decoder_vect) * self.c_vector

        # Add up the MSE loss + encoder and decoder regularization
        total = loss_mse + prior_ie + prior_id

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
