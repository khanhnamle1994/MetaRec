# Import utility scripts
from modules.input_loading import UserLoading, ItemLoading
from modules.info_embedding import UserEmbedding, ItemEmbedding
from modules.rec_model import RecMAM
from modules.memories import FeatureMem, TaskMem
from models import BaseModel, LocalUpdate, user_mem_init, maml_train
from configs import config_settings
from utils import train_test_user_list, default_info, grads_sum

# Import package
import torch


class MAMRec:
    """
    Initialize the MAMO Rec model class
    """

    def __init__(self):
        # Initialize variables
        self.support_size = config_settings['support_size']  # support size
        self.query_size = config_settings['query_size']  # query size
        self.n_epoch = config_settings['n_epoch']  # number of epochs
        self.n_inner_loop = config_settings['n_inner_loop']  # number of inner loop
        self.batch_size = config_settings['batch_size']  # batch size
        self.n_layer = config_settings['n_layer']  # number of layers
        self.embedding_dim = config_settings['embedding_dim']  # embedding dimension
        self.rho = config_settings['rho']  # local learning rate
        self.lamda = config_settings['lambda']  # global learning rate
        self.tao = config_settings['tao']  # hyper-parameter to control how much bias term is considered
        self.device = torch.device(config_settings['cuda_option'])
        self.n_k = config_settings['n_k']  # number of latent factors
        self.alpha = config_settings['alpha']  # hyper-parameter to control how much new profile info is added
        self.beta = config_settings['beta']  # hyper-parameter to control how much new information is kept
        self.gamma = config_settings['gamma']  # hyper-parameter to control how much new preference info is added
        self.active_func = config_settings['activation_function']  # choice of activation function
        self.rand = config_settings['rand']  # Boolean value to turn on randomization
        self.random_state = config_settings['random_state']  # Random seed state
        self.split_ratio = config_settings['split_ratio']  # train and test split ratio

        # Load dataset
        self.train_users, self.test_users = train_test_user_list(
            rand=self.rand, random_state=self.random_state, train_test_split_ratio=self.split_ratio
        )

        self.x1_loading, self.x2_loading = UserLoading(embedding_dim=self.embedding_dim).to(self.device), \
                                           ItemLoading(embedding_dim=self.embedding_dim).to(self.device)

        self.n_y = default_info['movielens']['n_y']

        # Create user and item embedding matrices
        self.UEmb = UserEmbedding(self.n_layer, default_info['movielens']['u_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)
        self.IEmb = ItemEmbedding(self.n_layer, default_info['movielens']['i_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)

        # Create recommendation model
        self.rec_model = RecMAM(self.embedding_dim, self.n_y, self.n_layer, activation=self.active_func).to(self.device)

        # Create the full model
        self.model = BaseModel(self.x1_loading, self.x2_loading, self.UEmb, self.IEmb, self.rec_model).to(self.device)

        # Collect task-specific user, model, and rating weights
        self.phi_u, self.phi_i, self.phi_r = self.model.get_weights()

        # Create the feature-specific memory component
        self.FeatureMem = FeatureMem(self.n_k, default_info['movielens']['u_in_dim'] * self.embedding_dim,
                                     self.model, device=self.device)

        # Create the task-specific memory component
        self.TaskMem = TaskMem(self.n_k, self.embedding_dim, device=self.device)

        # Train the model with meta optimization
        self.train = self.train_with_meta_optimization

        # Test the model with meta optimization
        self.test = self.test_with_meta_optimization

        # Train the model
        self.train()

        # Evaluate the model
        self.test()

    def train_with_meta_optimization(self):
        """
        Train the model with meta-optimization
        """
        # Loop through all epochs
        for i in range(self.n_epoch):
            # Get the model zero weights
            u_grad_sum, i_grad_sum, r_grad_sum = self.model.get_zero_weights()

            # Loop through up to 100 users in the training/support set
            for u in self.train_users[:100]:
                # Initialize local parameters (theta_u, theta_i, theta_r)
                bias_term, att_values = user_mem_init(u, self.device, self.FeatureMem, self.x1_loading, self.alpha)

                # Initialize user memory weights
                self.model.init_u_mem_weights(self.phi_u, bias_term, self.tao, self.phi_i, self.phi_r)

                # Initialize user-item memory weights
                self.model.init_ui_mem_weights(att_values, self.TaskMem)

                # Perform local update
                user_module = LocalUpdate(self.model, u, self.support_size, self.query_size, self.batch_size,
                                          self.n_inner_loop, self.rho, top_k=3, device=self.device)

                # Collect the user, item, and rating gradients
                u_grad, i_grad, r_grad = user_module.train()

                # Sum up the gradients
                u_grad_sum, i_grad_sum, r_grad_sum = grads_sum(u_grad_sum, u_grad), \
                                                     grads_sum(i_grad_sum, i_grad), \
                                                     grads_sum(r_grad_sum, r_grad)

                # Update the feature-specific memory parameters
                self.FeatureMem.write_head(u_grad, self.beta)

                # Get user-item memory weights
                u_mui = self.model.get_ui_mem_weights()

                # Update the task-specific memory parameters
                self.TaskMem.write_head(u_mui[0], self.gamma)

            # Train the model
            self.phi_u, self.phi_i, self.phi_r = maml_train(self.phi_u, self.phi_i, self.phi_r,
                                                            u_grad_sum, i_grad_sum, r_grad_sum, self.lamda)

            # Test the model
            self.test_with_meta_optimization()

    def test_with_meta_optimization(self):
        """
        Test the model with meta-optimization
        """
        # Collect the best performing task-specific user, item, and rating weights
        best_phi_u, best_phi_i, best_phi_r = self.model.get_weights()

        # Loop through all users in the test/query set
        for u in self.test_users:
            bias_term, att_values = user_mem_init(u, self.device, self.FeatureMem, self.x1_loading, self.alpha)
            self.model.init_u_mem_weights(best_phi_u, bias_term, self.tao, best_phi_i, best_phi_r)
            self.model.init_ui_mem_weights(att_values, self.TaskMem)
            self.model.init_weights(best_phi_u, best_phi_i, best_phi_r)

            # Perform local update
            user_module = LocalUpdate(self.model, u, self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.rho, top_k=3, device=self.device)
            user_module.test()


# Execute the model
if __name__ == '__main__':
    MAMRec()
