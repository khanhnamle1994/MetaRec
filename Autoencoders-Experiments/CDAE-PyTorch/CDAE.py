"""
Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
https://alicezheng.org/papers/wsdm16-cdae.pdf
"""
# Import packages and utility scripts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import utility scripts
from BaseModel import BaseModel
from Tools import apply_activation


class CDAE(BaseModel):
    """
    Collaborative Denoising Autoencoder model class
    """
    def __init__(self, model_conf, num_users, num_items, device):
        """
        :param model_conf: model configuration
        :param num_users: number of users
        :param num_items: number of items
        :param device: choice of device
        """
        super(CDAE, self).__init__()
        self.hidden_dim = model_conf.hidden_dim
        self.act = model_conf.act
        self.corruption_ratio = model_conf.corruption_ratio
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        self.to(self.device)

    def forward(self, user_id, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        # normalize the rating matrix
        user_degree = torch.norm(rating_matrix, 2, 1).view(-1, 1)  # user, 1
        item_degree = torch.norm(rating_matrix, 2, 0).view(1, -1)  # 1, item
        normalize = torch.sqrt(user_degree @ item_degree)
        zero_mask = normalize == 0
        normalize = torch.masked_fill(normalize, zero_mask.bool(), 1e-10)

        normalized_rating_matrix = rating_matrix / normalize

        # corrupt the rating matrix
        normalized_rating_matrix = F.dropout(normalized_rating_matrix, self.corruption_ratio, training=self.training)

        # build the collaborative denoising autoencoder
        enc = self.encoder(normalized_rating_matrix) + self.user_embedding(user_id)
        enc = apply_activation(self.act, enc)
        dec = self.decoder(enc)

        return torch.sigmoid(dec)

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        """
        Train model for one epoch
        :param dataset: given data
        :param optimizer: choice of optimizer
        :param batch_size: batch size
        :param verbose: verbose
        :return: model loss
        """
        self.train()

        # user, item, rating pairs
        train_matrix = dataset.train_matrix

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
            batch_idx = torch.LongTensor(batch_idx).to(self.device)
            pred_matrix = self.forward(batch_idx, batch_matrix)

            # cross_entropy
            batch_loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='sum')
            # batch_loss = batch_matrix * (pred_matrix + 1e-10).log() + (1 - batch_matrix) * (1 - pred_matrix + 1e-10).log()
            # batch_loss = -torch.sum(batch_loss)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, eval_users, eval_pos, test_batch_size):
        """
        Predict the model on test set
        :param eval_users: evaluation (test) user
        :param eval_pos: position of the evaluated (test) item
        :param test_batch_size: batch size for test set
        :return: predictions
        """
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray()).to(self.device)
            preds = np.zeros_like(input_matrix)

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]

                test_batch_matrix = input_matrix[batch_idx]
                batch_idx = torch.LongTensor(batch_idx).to(self.device)
                batch_pred_matrix = self.forward(batch_idx, test_batch_matrix)
                batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        return preds
