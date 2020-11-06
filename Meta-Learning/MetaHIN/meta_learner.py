# Import packages
import torch
from torch.nn import functional as F


class MetaLearner(torch.nn.Module):
    """
    Construct a Meta-Learner class
    """
    def __init__(self, config):
        """
        Initialize the Meta-Learner class
        :param config: experiment configuration
        """
        super(MetaLearner, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = 32 + config['item_embedding_dim']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.config = config

        # prediction parameters
        self.vars = torch.nn.ParameterDict()
        self.vars_bn = torch.nn.ParameterList()

        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim, self.fc1_in_dim]))  # 64, 96
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars['ml_fc_w3'] = w3
        self.vars['ml_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_emb, item_emb, user_neigh_emb, vars_dict=None):
        """
        Perform a forward pass
        :param user_emb: user embedding
        :param item_emb: item embedding
        :param user_neigh_emb: user neighbor embeddings
        :param vars_dict: variable dictionary
        """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_neigh_emb

        x = torch.cat((x_i, x_u), 1)  # ?, item_emb_dim+user_emb_dim+user_emb_dim
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        """
        Zero out the gradients
        :param vars_dict: variable dictionary
        """
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        """
        Update the parameters
        """
        return self.vars


class MetapathLearner(torch.nn.Module):
    """
    Construct a Metapath Learner class
    """
    def __init__(self, config):
        super(MetapathLearner, self).__init__()
        self.config = config

        # meta-path parameters
        self.vars = torch.nn.ParameterDict()
        neigh_w = torch.nn.Parameter(torch.ones([32, config['item_embedding_dim']]))  # dim=32, movielens 0.81006
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, user_emb, item_emb, neighs_emb, mp, index_list, vars_dict=None):
        """
        Perform a forward pass
        :param user_emb: user embedding
        :param item_emb: item embedding
        :param neighs_emb: neighbor embedding
        :param mp: meta-path
        :param index_list: index list
        :param vars_dict: variable dictionary
        """
        if vars_dict is None:
            vars_dict = self.vars

        # (#neighbors, item_emb_dim)
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        # (#sample, user_emb_dim)
        output_emb = F.leaky_relu(torch.mean(agg_neighbor_emb, 0)).repeat(user_emb.shape[0], 1)

        return output_emb

    def zero_grad(self, vars_dict=None):
        """
        Zero out the gradients
        :param vars_dict: variable dictionary
        """
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        """
        Update the parameters
        """
        return self.vars
