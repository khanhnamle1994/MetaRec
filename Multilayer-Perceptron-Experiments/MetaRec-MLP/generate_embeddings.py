# Import PyTorch package
import torch


class item(torch.nn.Module):
    """
    Item class
    """
    def __init__(self, config):
        """
        Initialize the item class
        :param config: experiment configuration
        """
        super(item, self).__init__()
        # Number of rate levels
        self.num_rate = config.num_rate
        # Number of genres
        self.num_genre = config.num_genre
        # Number of directors
        self.num_director = config.num_director
        # Number of actors
        self.num_actor = config.num_actor
        # Number of embedding dimensions
        self.embedding_dim = config.embedding_dim

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

        # Create director embeddings
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        # Create actor embeddings
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        """
        Perform forward pass on movie items
        :param rate_idx: Index of the rate category
        :param genre_idx: Index of the genre
        :param director_idx: Index of the director name
        :param actors_idx: Index of the actor
        :param vars: Other variables
        :return: one-dimensional embedding
        """
        # Perform embedding processes based on the input
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)

        # Concatenate the embedded vectors
        return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)


class user(torch.nn.Module):
    """
    User class
    """
    def __init__(self, config):
        """
        Initialize the user class
        :param config: experiment configuration
        """
        super(user, self).__init__()
        # Number of genders
        self.num_gender = config.num_gender
        # Number of ages
        self.num_age = config.num_age
        # Number of occupations
        self.num_occupation = config.num_occupation
        # Number of zipcodes
        self.num_zipcode = config.num_zipcode
        # Number of embedding dimensions
        self.embedding_dim = config.embedding_dim

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

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        """
        Perform forward pass on user information
        :param gender_idx: Index of the gender
        :param age_idx: Index of the age
        :param occupation_idx: Index of the occupation
        :param area_idx: Index of the zipcode area
        :return: one-dimensional embedding
        """
        # Perform embedding processes based on the input
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)

        # Concatenate the embedded vectors
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
