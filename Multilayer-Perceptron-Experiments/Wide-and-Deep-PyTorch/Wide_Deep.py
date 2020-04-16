import torch

from layer import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron


class WideAndDeepModel(torch.nn.Module):
    """
    A Pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        """
        :param field_dims: Number of input dimensions
        :param embed_dim: Number of dense embedding dimensions
        :param mlp_dims: Number of hidden layers
        :param dropout: dropout rate
        """
        super().__init__()

        # Wide Learning Component
        self.linear = FeaturesLinear(field_dims)

        # Deep Learning Component
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Get feature embeddings
        embed_x = self.embedding(x)

        # Joint learning of the wide component and the deep component
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))