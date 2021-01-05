# Import libraries
import torch
from torch.nn import functional as F
import torch.nn as nn

# Import utility script
from generate_embeddings import item, user


class Linear(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        """
        Initialize a linear model
        :param in_features: input features
        :param out_features: output features
        """
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        """
        Perform a forward pass
        :param x: input
        :return: output
        """
        if self.weight.fast is not None and self.bias.fast is not None:
            # fast weight is the temporarily adapted weight
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear, self).forward(x)
        return out


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        """
        Initialize the user preference estimation model class
        :param config: experiment configuration
        """
        super(user_preference_estimator, self).__init__()

        # Specify the input and output dimensions
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 8
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim

        # Specify the item and user embeddings
        self.item_emb = item(config)
        self.user_emb = user(config)

        # Specify the decision-making fully-connected layers
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)

        # Specify the linear output layer
        self.linear_out = Linear(self.fc2_out_dim, 1)

        # Specify the final output layer
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)

    def forward(self, x, training=True):
        """
        Perform a forward pass
        :param x: item and user input
        :param training: boolean training mode
        :return: output
        """
        rate_idx = x[:, 0]
        genre_idx = x[:, 1:26]
        director_idx = x[:, 26:2212]
        actor_idx = x[:, 2212:10242]
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]

        # Performs embedding processes based on the item and user input
        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)

        # Concatenates the item and user embeddings
        x = torch.cat((item_emb, user_emb), 1)

        # Returns the final output
        x = self.final_part(x)
        return x
