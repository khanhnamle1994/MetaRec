import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
# from utils import use_cuda
from utils import resume_checkpoint


class NeuMF(torch.nn.Module):

    def __init__(self, config):
        """
        Function to initialize the NMF class
        :param config: configuration choice
        """
        super(NeuMF, self).__init__()

        self.config = config
        # Specify number of users and number of items
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        # Specify number of latent dimensions for both pretrained GMF and MLP models
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        # Generate user and item embedding for MLP model
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        # Generate user and item embedding for GMF model
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # Generate a list of Fully-Connected layers
        self.fc_layers = torch.nn.ModuleList()
        # Apply linear transformations between each fully-connected layer
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Apply a linear transformation to the incoming last fully-connected layer -> output of size 1
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        # Perform sigmoid activation function
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """
        Function to perform a forward pass for rating prediction
        :param user_indices: a list of user indices
        :param item_indices: a list of item indices
        :return: predicted rating
        """

        # Generate user and item embedding from user indices and item indices for MLP model
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        # Generate user and item embedding from user indices and item indices for GMF model
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # Concatenate the user embedding and item embedding values to get the MLP vector
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        # Multiply the user embedding and item embedding values to get the MF vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        # Go through all fully-connected layers
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            # Perform ReLU activation
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        # Concatenate the MLP vector and MF vector to get the final vector
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        # Apply linear transformation to the final vector
        logits = self.affine_output(vector)
        # Apply sigmoid (logistic) to get the final predicted rating
        rating = self.logistic(logits)

        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""

        config = self.config
        config['latent_dim'] = config['latent_dim_mlp']
        mlp_model = MLP(config)

        # if config['use_cuda'] is True:
        #     mlp_model.cuda()

        # resume_checkpoint(mlp_model, model_dir = config['pretrain_mlp'], device_id = config['device_id'])
        resume_checkpoint(mlp_model, model_dir=config['pretrain_mlp'])

        # Get the user and item weights from the trained MLP model
        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data

        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        config['latent_dim'] = config['latent_dim_mf']
        gmf_model = GMF(config)

        # if config['use_cuda'] is True:
        #     gmf_model.cuda()

        # resume_checkpoint(gmf_model, model_dir = config['pretrain_mf'], device_id = config['device_id'])
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'])

        # Get the user and item weights from the trained GMF model
        self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        # Perform linear transformation to get the final weight and bias values from both MLP and GMF weights
        self.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)


class NeuMFEngine(Engine):
    """Engine for training & evaluating NMF model"""

    def __init__(self, config):
        self.model = NeuMF(config)

        # if config['use_cuda'] is True:
        #     use_cuda(True, config['device_id'])
        #     self.model.cuda()

        super(NeuMFEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()
