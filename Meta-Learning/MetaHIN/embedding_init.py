# Import libraries
import torch
from torch.autograd import Variable


class UserEmbedding(torch.nn.Module):
    """
    Initialize user embedding class
    """
    def __init__(self, config):
        """
        Initialize the user class
        :param config: experiment configuration
        """
        super(UserEmbedding, self).__init__()
        self.num_gender = config['num_gender']  # Number of genders
        self.num_age = config['num_age']  # Number of ages
        self.num_occupation = config['num_occupation']  # Number of occupations
        self.num_zipcode = config['num_zipcode']  # Number of zipcodes

        self.embedding_dim = config['embedding_dim']  # Number of embedding dimensions

        # Create gender embeddings
        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        # Create age embeddings
        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        # Create occupation embeddings
        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        # Create zipcode area embeddings
        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        Perform forward pass on user features
        :param user_fea: user features
        :return: one-dimensional embedding
        """
        # Collect user features
        gender_idx = Variable(user_fea[:, 0], requires_grad=False)
        age_idx = Variable(user_fea[:, 1], requires_grad=False)
        occupation_idx = Variable(user_fea[:, 2], requires_grad=False)
        area_idx = Variable(user_fea[:, 3], requires_grad=False)

        # Perform embedding processes based on the user input
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)

        # Concatenate the embedded vectors
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)  # (1, 4*32)


class ItemEmbeddingML(torch.nn.Module):
    """
    Initialize item embedding class
    """
    def __init__(self, config):
        """
        Initialize the item class
        :param config: experiment configuration
        """
        super(ItemEmbeddingML, self).__init__()
        self.num_rate = config['num_rate']  # Number of rate levels
        self.num_genre = config['num_genre']  # Number of genres
        self.embedding_dim = config['embedding_dim']  # Number of embedding dimensions

        # Create rate category embeddings
        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        # Create genre embeddings
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, item_fea):
        """
        Perform forward pass on item features
        :param item_fea: item features
        :return: one-dimensional embedding
        """
        # Collect item features
        rate_idx = Variable(item_fea[:, 0], requires_grad=False)
        genre_idx = Variable(item_fea[:, 1:26], requires_grad=False)

        # Perform embedding processes based on the item input
        rate_emb = self.embedding_rate(rate_idx)  # (1,32)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)  # (1,32)

        # Concatenate the embedded vectors
        return torch.cat((rate_emb, genre_emb), 1)  # (1, 2*32)
