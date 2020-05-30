"""
Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019.
https://arxiv.org/pdf/1905.03375
"""
# Import packages
import numpy as np
import torch

# Import utility script
from BaseModel import BaseModel


class ESAE(BaseModel):
    """
    Embarrassingly Shallow Autoencoders model class
    """
    def __init__(self, model_conf, num_users, num_items, device):
        """
        :param model_conf: model configuration
        :param num_users: number of users
        :param num_items: number of items
        :param device: choice of device
        """
        super(ESAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.reg = model_conf.reg

        self.device = device
        self.to(self.device)

    def forward(self, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        G = rating_matrix.transpose(0, 1) @ rating_matrix

        diag = list(range(G.shape[0]))
        G[diag, diag] += self.reg
        P = G.inverse()

        # B = P * (X^T * X − diagMat(γ))
        self.enc_w = P / -torch.diag(P)
        min_dim = min(*self.enc_w.shape)
        self.enc_w[range(min_dim), range(min_dim)] = 0

        # Calculate the output matrix for prediction
        output = rating_matrix @ self.enc_w

        return output

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

        # Solve ESAE
        train_matrix = torch.FloatTensor(dataset.train_matrix.toarray()).to(self.device)
        output = self.forward(train_matrix)

        loss = 0.0

        return loss

    def generate_mask(self, mask_shape):
        return self.binomial.sample(mask_shape).to(self.device)

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
                batch_pred_matrix = (test_batch_matrix @ self.enc_w)
                batch_pred_matrix.masked_fill(test_batch_matrix.bool(), float('-inf'))
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        return preds