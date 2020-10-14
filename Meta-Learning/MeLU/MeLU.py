# Import packages
import torch
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

# Import utility script
from embeddings import item, user


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        """
        Initialize the user preference estimation model class
        :param config: experiment configuration
        """
        super(user_preference_estimator, self).__init__()

        # Specify the input and output dimensions
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']

        # Specify the item and user embeddings
        self.item_emb = item(config)
        self.user_emb = user(config)

        # Specify the decision-making fully-connected layers
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)

        # Specify the linear output layer
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training=True):
        """
        Perform a forward pass
        :param x: item and user input
        :param training: boolean training mode
        :return: output
        """
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        # Performs embedding processes based on the item and user input
        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)

        # Concatenates the item and user embeddings
        x = torch.cat((item_emb, user_emb), 1)

        # Uses two fully-connected layers with ReLU activation function
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Linear function at the output layer
        return self.linear_out(x)


class MeLU(torch.nn.Module):
    def __init__(self, config):
        """
        Initialize the MeLU class
        :param config: experiment configuration
        """
        super(MeLU, self).__init__()

        # Initialize the user preference estimation model
        self.model = user_preference_estimator(config)

        # Specify the local learning rate
        self.local_lr = config['local_lr']
        self.store_parameters()

        # Specify Adam optimization
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

        # Specify weight names during local update
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias',
                                                'fc2.weight', 'fc2.bias',
                                                'linear_out.weight', 'linear_out.bias']

    def store_parameters(self):
        """
        Store the model parameters
        """
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        """
        Perform a forward pass
        :param support_set_x: features of the support set data
        :param support_set_y: target of the support set data
        :param query_set_x: features of the query set data
        :param num_local_update: number of local updates to be performed
        """
        # Loop through number of local updates
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())

            # Predicted target of the support set
            support_set_y_pred = self.model(support_set_x)
            # Calculate MSE loss between predicted and ground-truth target of the support set
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # Back-propagate the gradient
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            # Perform a local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]

        self.model.load_state_dict(self.fast_weights)
        # Predict the target of the query set
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)

        # Return the predicted target of the query set
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        """
        Perform a global update
        :param support_set_xs: number of features of the support set data
        :param support_set_ys: number of targets of the support set data
        :param query_set_xs: number of features of the query set data
        :param query_set_ys: number of targets of the query set data
        :param num_local_update: number of local updates to be performed
        """
        batch_sz = len(support_set_xs)
        losses_q = []

        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
        return

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        """
        Get the average Frobenius norm of the users' gradient for personalization
        """
        tmp = 0.
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

            for i in range(self.weight_len):
                # For averaging Frobenius norm
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / num_local_update
