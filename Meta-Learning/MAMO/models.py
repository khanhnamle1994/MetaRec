# Import package
import torch
import pickle

# Import utility scripts
from utils import get_params, get_zeros_like_params, init_params, get_grad, init_u_mem_params,\
    init_ui_mem_params, load_user_info, update_parameters, UserDataLoader, to_torch, mae, ndcg
from torch.utils.data import DataLoader


class BaseModel(torch.nn.Module):
    """
    Initialize a base model class
    """
    def __init__(self, input1_module, input2_module, embedding1_module, embedding2_module, rec_module):
        super(BaseModel, self).__init__()

        self.input_user_loading = input1_module
        self.input_item_loading = input2_module
        self.user_embedding = embedding1_module
        self.item_embedding = embedding2_module
        self.rec_model = rec_module

    def forward(self, x1, x2):
        """
        Perform a forward pass
        :param x1: User input data
        :param x2: Item input data
        :return: Rating score
        """
        # Initial user profile (pu) and item profile (pi)
        pu, pi = self.input_user_loading(x1), self.input_item_loading(x2)
        # Learn the user embedding (eu) and item embedding (ei)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        # Given the user and item embeddings, get the prediction of the preference score
        rec_value = self.rec_model(eu, ei)
        return rec_value

    def get_weights(self):
        """
        Collect the parameters from user embedding, item embedding, and recommendation modules
        """
        # User embedding parameters
        u_emb_params = get_params(self.user_embedding.parameters())
        # Item embedding parameters
        i_emb_params = get_params(self.item_embedding.parameters())
        # Recommendation model parameters
        rec_params = get_params(self.rec_model.parameters())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        """
        Collect zero-like parameters from user embedding, item embedding, and recommendation modules
        """
        # User embedding zero-like parameters
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.parameters())
        # Item embedding zero-like parameters
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.parameters())
        # Recommendation model zero-like parameters
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.parameters())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        """
        Initialize parameters
        :param u_emb_para: User embedding parameters
        :param i_emb_para: Item embedding parameters
        :param rec_para: Recommendation parameters
        """
        # Initialize user embedding parameters
        init_params(self.user_embedding.parameters(), u_emb_para)
        # Initialize item embedding parameters
        init_params(self.item_embedding.parameters(), i_emb_para)
        # Initialize recommendation model parameters
        init_params(self.rec_model.parameters(), rec_para)

    def get_grad(self):
        """
        Collect the user-specific, item-specific, and rating-specific gradients
        """
        # Collect user-specific gradients from the user embedding parameters
        u_grad = get_grad(self.user_embedding.parameters())
        # Collect item-specific gradients from the item embedding parameters
        i_grad = get_grad(self.item_embedding.parameters())
        # Collect rating-specific gradients from the recommendation model parameters
        r_grad = get_grad(self.rec_model.parameters())
        return u_grad, i_grad, r_grad

    def init_u_mem_weights(self, u_emb_para, mu, tao, i_emb_para, rec_para):
        init_u_mem_params(self.user_embedding.parameters(), u_emb_para, mu, tao)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def init_ui_mem_weights(self, att_values, task_mem):
        # Initialize the weights only for the memory layer
        u_mui = task_mem.read_head(att_values)
        init_ui_mem_params(self.rec_model.mem_layer.parameters(), u_mui)

    def get_ui_mem_weights(self):
        return get_params(self.rec_model.mem_layer.parameters())


class LocalUpdate:
    def __init__(self, your_model, u_idx, sup_size, que_size, bt_size, n_loop, update_lr, top_k, device):
        # Load user information
        self.s_x1, self.s_x2, self.s_y, self.s_y0, self.q_x1, self.q_x2, self.q_y, self.q_y0 = load_user_info(
            u_idx, sup_size, que_size, device
        )

        # Load user data
        user_data = UserDataLoader(self.s_x1, self.s_x2, self.s_y, self.s_y0)
        self.user_data_loader = DataLoader(user_data, batch_size=bt_size)

        # Specify model
        self.model = your_model

        # Specify learning rate and optimizer
        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        # Specify loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Specify number of loops and number of ranked items
        self.n_loop = n_loop
        self.top_k = top_k

        # Specify the device
        self.device = device
        self.s_x1, self.s_x2, self.s_y = self.s_x1.to(self.device), self.s_x2.to(self.device), self.s_y.to(self.device)
        self.q_x1, self.q_x2, self.q_y = self.q_x1.to(self.device), self.q_x2.to(self.device), self.q_y.to(self.device)

    def train(self):
        """
        Perform training
        :return: user-specific, item-specific, and rating-specific gradients
        """
        for i in range(self.n_loop):
            # Train the model on the support set
            for i_batch, (x1, x2, y, y0) in enumerate(self.user_data_loader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                # Predict the ratings on the support set
                pred_y = self.model(x1, x2)
                # Calculate the loss
                loss = self.loss_fn(pred_y, y)
                # Set the gradients to 0
                self.optimizer.zero_grad()
                # Update the local task-specific parameter
                loss.backward()
                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                # Perform a parameter update based on the current gradient and the update rule
                self.optimizer.step()

        # Predict the ratings on the query set
        q_pred_y = self.model(self.q_x1, self.q_x2)
        # Set the gradients to 0
        self.optimizer.zero_grad()
        # Calculate the loss
        loss = self.loss_fn(q_pred_y, self.q_y)
        # Update the local task-specific parameters
        loss.backward()
        # Perform gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        # Collect the gradients
        u_grad, i_grad, r_grad = self.model.get_grad()
        return u_grad, i_grad, r_grad

    def test(self):
        """
        Perform testing
        """
        for i in range(self.n_loop):
            # Test the model on the support set
            for i_batch, (x1, x2, y, y0) in enumerate(self.user_data_loader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                # Predict the ratings on the support set
                pred_y = self.model(x1, x2)
                # Calculate the loss
                loss = self.loss_fn(pred_y, y)
                # Set the gradients to 0
                self.optimizer.zero_grad()
                # Update the local task-specific parameters
                loss.backward()
                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                # Perform a parameter update based on the current gradient and the update rule
                self.optimizer.step()

        # Calculate the predicted ratings on the query set
        q_pred_y = self.model(self.q_x1, self.q_x2)
        # Calculate the MAE
        mean_absolute_error = mae(self.q_y, q_pred_y)
        # Calculate the NDCG
        NDCG = ndcg(self.q_y, q_pred_y)

        print("MAE: ", mean_absolute_error)
        print("NDCG: ", NDCG)
        return mean_absolute_error, NDCG


def maml_train(raw_phi_u, raw_phi_i, raw_phi_r, u_grad_list, i_grad_list, r_grad_list, global_lr):
    """
    Update the global model parameters
    :param raw_phi_u: global user parameter
    :param raw_phi_i: global item parameter
    :param raw_phi_r: global rating parameter
    :param u_grad_list: list of user gradients
    :param i_grad_list: list of item gradients
    :param r_grad_list: list of rating gradients
    :param global_lr: global learning rate
    """
    phi_u = update_parameters(raw_phi_u, u_grad_list, global_lr)
    phi_i = update_parameters(raw_phi_i, i_grad_list, global_lr)
    phi_r = update_parameters(raw_phi_r, r_grad_list, global_lr)
    return phi_u, phi_i, phi_r


def user_mem_init(u_id, device, feature_mem, loading_model, alpha):
    """
    Initialize user memory cube with personalized bias term and attention values
    :param u_id: User ID
    :param device: Device choice
    :param feature_mem: Feature-specific memory component
    :param loading_model: Loaded model
    :param alpha: Hyper-parameter
    :return: Personalized bias term and attention values
    """
    # Path to raw processed data (in Pickle files)
    path = 'data_prep/processed_data/raw/'
    # Load the Pickle files
    u_x1_data = pickle.load(open('{}sample_{}_x1.p'.format(path, str(u_id)), 'rb'))
    # Convert the user data into PyTorch tensor
    u_x1 = to_torch([u_x1_data]).to(device)
    # Get user profile matrix
    pu = loading_model(u_x1)
    # Retrieve the personalized bias term and the attention values
    personalized_bias_term, att_values = feature_mem.read_head(pu, alpha)
    # Delete variables to save storage
    del u_x1_data, u_x1, pu
    return personalized_bias_term, att_values
