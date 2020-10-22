# Import library
import torch
import torch.nn.functional as F

# Import utility script
from utils import config


class UserLoading(torch.nn.Module):
    """
    Create a class to load user input
    """

    def __init__(self, embedding_dim):
        """
        Initialize the class
        :param embedding_dim: the embedding dimension
        """
        super(UserLoading, self).__init__()
        # Load user-specific features
        self.gender_dim = config['n_gender']
        self.age_dim = config['n_age']
        self.occupation_dim = config['n_occupation']

        # Construct user-specific feature embeddings
        self.embedding_dim = embedding_dim
        self.embedding_gender = torch.nn.Embedding(num_embeddings=self.gender_dim,
                                                   embedding_dim=self.embedding_dim)
        self.embedding_age = torch.nn.Embedding(num_embeddings=self.age_dim,
                                                embedding_dim=self.embedding_dim)
        self.embedding_occupation = torch.nn.Embedding(num_embeddings=self.occupation_dim,
                                                       embedding_dim=self.embedding_dim)

    def forward(self, x1):
        """
        Perform a forward pass
        :param x1: User Info
        :return: Concatenated embedding with all the user-specific feature embeddings
        """
        # Collect the gender, age, and occupation indexes
        gender_idx, age_idx, occupation_idx = x1[:, 0], x1[:, 1], x1[:, 2]
        # Construct gender embedding
        gender_emb = self.embedding_gender(gender_idx)
        # Construct age embedding
        age_emb = self.embedding_age(age_idx)
        # Construct occupation embedding
        occupation_emb = self.embedding_occupation(occupation_idx)
        # Concatenate the embeddings above
        concat_emb = torch.cat((gender_emb, age_emb, occupation_emb), 1)

        return concat_emb


class ItemLoading(torch.nn.Module):
    """
    Create a class to load item input
    """
    def __init__(self, embedding_dim):
        """
        Initialize the class
        :param embedding_dim: the embedding dimension
        """
        super(ItemLoading, self).__init__()
        # Load item-specific features
        self.rate_dim = config['n_rate']
        self.genre_dim = config['n_genre']
        self.director_dim = config['n_director']
        self.year_dim = config['n_year']

        # Construct item-specific feature embeddings
        self.embedding_dim = embedding_dim
        self.embedding_rate = torch.nn.Embedding(num_embeddings=self.rate_dim,
                                                 embedding_dim=self.embedding_dim)
        self.embedding_genre = torch.nn.Linear(in_features=self.genre_dim,
                                               out_features=self.embedding_dim, bias=False)
        self.embedding_director = torch.nn.Linear(in_features=self.director_dim,
                                                  out_features=self.embedding_dim, bias=False)
        self.embedding_year = torch.nn.Embedding(num_embeddings=self.year_dim,
                                                 embedding_dim=self.embedding_dim)

    def forward(self, x2):
        """
        Perform a forward pass
        :param x2: Item Info
        :return: Concatenated embedding with all the item-specific feature embeddings
        """
        # Collect the rate, year, genre, and director indexes
        rate_idx, year_idx, genre_idx, director_idx = x2[:, 0], x2[:, 1], x2[:, 2:27], x2[:, 27:]
        # Construct rate embeddings
        rate_emb = self.embedding_rate(rate_idx)
        # Construct year embeddings
        year_emb = self.embedding_year(year_idx)
        # Construct genre embeddings
        genre_emb = F.sigmoid(self.embedding_genre(genre_idx.float()))
        # Construct director embeddings
        director_emb = F.sigmoid(self.embedding_director(director_idx.float()))
        # Concatenate the embeddings above
        concat_emb = torch.cat((rate_emb, year_emb, genre_emb, director_emb), 1)

        return concat_emb