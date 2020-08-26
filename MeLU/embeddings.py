# Import PyTorch libraries
import torch


# Item class
class item(torch.nn.Module):
    def __init__(self, config):
        super(item, self).__init__()
        # Number of genres
        self.num_genre = config['num_genre']
        # Number of embedding dimensions
        self.embedding_dim = config['embedding_dim']

        # Create genre embeddings
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, genre_idx, vars=None):
        """
        Perform forward pass on movie items
        :param genre_idx: Index of the genre
        :param vars: Other variables
        """
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        return torch.cat((genre_emb), 1)


# User class
class user(torch.nn.Module):
    def __init__(self, config):
        super(user, self).__init__()
        # Number of genders
        self.num_gender = config['num_gender']
        # Number of ages
        self.num_age = config['num_age']
        # Number of occupations
        self.num_occupation = config['num_occupation']
        # Number of zipcodes
        self.num_zipcode = config['num_zipcode']
        # Number of embedding dimensions
        self.embedding_dim = config['embedding_dim']

        # Create gender embeddings
        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        # Create genre embeddings
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

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        """
        Perform forward pass on user profiles
        :param gender_idx: Index of the gender
        :param age_idx: Index of the age
        :param occupation_idx: Index of the occupation
        :param area_idx: Index of the zipcode area
        """
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
