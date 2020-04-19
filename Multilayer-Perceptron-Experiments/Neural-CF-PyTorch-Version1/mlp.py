import torch
from gmf import GMF
from engine import Engine
# from utils import use_cuda
from utils import resume_checkpoint


class MLP(torch.nn.Module):

    def __init__(self, config):
        """
        Function to initialize the MLP class
        :param config: configuration choice
        """
        super(MLP, self).__init__()

        self.config = config
        # Specify number of users, number of items, and number of latent dimensions
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        # Generate user embedding
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # Generate item embedding
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Generate a list of Fully-Connected layers
        self.fc_layers = torch.nn.ModuleList()
        # Apply linear transformations between each fully-connected layer
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Apply a linear transformation to the incoming last fully-connected layer -> output of size 1
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
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
        # Concatenate the user and iem embeddings -> Resulting a latent vector
        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        # Go through all fully-connected layers
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            # Perform ReLU activation
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)

        # Apply linear transformation to the final vector
        logits = self.affine_output(vector)
        # Apply sigmoid (logistic) to get the final predicted rating
        rating = self.logistic(logits)

        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""

        config = self.config
        gmf_model = GMF(config)

        # if config['use_cuda'] is True:
        #     gmf_model.cuda()

        # resume_checkpoint(gmf_model, model_dir = config['pretrain_mf'], device_id = config['device_id'])
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'])

        # Get the user weights from the trained GMF model
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        # Get the item weights from the trained GMF model
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = MLP(config)

        # if config['use_cuda'] is True:
        #     use_cuda(True, config['device_id'])
        #     self.model.cuda()

        super(MLPEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()
