# Import PyTorch Packages
import torch
from torch import nn
import torch.nn.functional as F


class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, n_user, n_item, n_occu, k=10, c_vector=1.0, c_bias=1.0, writer=None):
        """
        :param n_user: User column
        :param n_item: Item column
        :param n_occu: Occupation column
        :param k: Dimensions constant
        :param c_vector: Regularization constant
        :param c_bias: Regularization constant for the biases
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.n_occu = n_occu
        self.c_bias = c_bias
        self.c_vector = c_vector

        # The embedding matrices for user and item are learned and fit by PyTorch
        self.user = nn.Embedding(n_user, k)
        self.item = nn.Embedding(n_item, k)

        # We've added new term here: Embedding matrices for occupation vectors
        self.occu = nn.Embedding(n_occu, k)

        # Embedding matrices for the user's biases and the item's biases
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)

        # Initialize the bias tensors
        self.bias = nn.Parameter(torch.ones(1))

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

        # The user-item interaction: p_u * q_i is a dot product between the user vector and the item vector
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)

        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)

        # These are the occupation indices, and correspond to "o" variable
        occu_id = train_x[:, 3]
        # Initialize a vector occupation = r_o using the occupation indices
        vector_occu = self.occu(occu_id)
        # The user-occupation interaction: p_u * r_o is a dot product between the user vector and the occupation vector
        uo_interaction = torch.sum(vector_user * vector_occu, dim=1)

        # Add the bias, the user-item interaction, and the user-occupation interaction to obtain the final prediction
        prediction = ui_interaction + uo_interaction + biases
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

        # Compute L2 regularization over user (P) and item (Q) matrices
        prior_user = l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector

        # Compute L2 regularization over occupation (R) matrices
        prior_occu = l2_regularize(self.occu.weight) * self.c_vector

        # Add up the MSE loss + user & item regularization + user & item biases regularization + occupation
        # regularization
        total = loss_mse + prior_user + prior_item + prior_bias_item + prior_bias_user + prior_occu

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
