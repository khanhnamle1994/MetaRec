# Import PyTorch Packages
import torch
from torch import nn
import torch.nn.functional as F


# Define the MF Model
class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, k=10, c_vector=1.0, writer=None):
        """
        :param n_user: User column
        :param n_item: Item column
        :param k: Dimensions constant
        :param c_vector: Regularization constant
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_vector = c_vector

        # The embedding matrices for user and item are learned and fit by PyTorch
        self.user = nn.Embedding(n_user, k)
        self.item = nn.Embedding(n_item, k)

    def __call__(self, train_x):
        """This is the most important function in this script"""
        # These are the user indices, and correspond to "u" variable
        user_id = train_x[:, 0]
        # These are the item indices, correspond to the "i" variable
        item_id = train_x[:, 1]

        # Initialize a vector user = p_u using the user indices
        vector_user = self.user(user_id)
        # Initialize a vector item = q_i using the item indices
        vector_item = self.item(item_id)

        # The user-item interaction: p_u * q_i is a dot product between the 2 vectors above
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)
        return ui_interaction

    def loss(self, prediction, target):
        """
        Function to calculate the loss metric
        """
        # Calculate the Mean Squared Error between target = R_ui and prediction = p_u * q_i
        loss_mse = F.mse_loss(prediction, target.squeeze())

        # Compute L2 regularization over user (P) and item (Q) matrices
        prior_user = l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector

        # Add up the MSE loss + user & item regularization
        total = loss_mse + prior_user + prior_item

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
