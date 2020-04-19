import torch
from engine import Engine


# from utils import use_cuda

class GMF(torch.nn.Module):

    def __init__(self, config):
        """
        Function to initialize the GMF class
        :param config: configuration choice
        """
        super(GMF, self).__init__()

        # Specify number of users, number of items, and number of latent dimensions
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        # Generate user embedding
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # Generate item embedding
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Apply a linear transformation to the incoming latent dimension -> output of size 1
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        # Perform sigmoid activation function
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """
        Function to perform a forward pass for rating prediction
        :param user_indices: a list of user indices
        :param item_indices: a list of item indices
        :return: predicted rating
        """
        # Generate user embedding from user indices
        user_embedding = self.embedding_user(user_indices)
        # Generate item embedding from item indices
        item_embedding = self.embedding_item(item_indices)

        # Perform dot product between user and item embedding values
        element_product = torch.mul(user_embedding, item_embedding)
        # Apply linear transformation to the dot product
        logits = self.affine_output(element_product)
        # Apply sigmoid (logistic) to get the final predicted rating
        rating = self.logistic(logits)

        return rating

    def init_weight(self):
        pass


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = GMF(config)
        # if config['use_cuda'] is True:
        #     use_cuda(True, config['device_id'])
        #     self.model.cuda()

        super(GMFEngine, self).__init__(config)
