# Import library
import torch

# Import utility script
from utils import activation_func


class FeatureMem:
    """
    Create a class to initialize the feature-specific memory component
    """
    def __init__(self, n_k, u_emb_dim, base_model, device):
        # Number of user preference types (pre-defined before training)
        self.n_k = n_k
        # Choice of base model
        self.base_model = base_model
        # User profile memory
        self.p_memory = torch.rand(n_k, u_emb_dim, device=device).normal_()  # on device
        u_param, _, _ = base_model.get_weights()
        self.u_memory = []

        for i in range(n_k):
            bias_list = []
            for param in u_param:
                bias_list.append(param.normal_(std=0.05))
            self.u_memory.append(bias_list)

        # Initialized attention values
        self.att_values = torch.zeros(n_k).to(device)
        # Choice of device
        self.device = device

    def read_head(self, p_u, alpha, train=True):
        """
        Read head to retrieve the memory
        :param p_u: user profile
        :param alpha: hyper-parameter to control how much new profile information is added
        :param train: boolean value to toggle training mode on/off
        :return: personalized bias term and attention values
        """
        # Initialize an Attention class
        att_model = Attention(self.n_k).to(self.device)
        # Collect the attention values
        attention_values = att_model(p_u, self.p_memory).to(self.device)
        # Get personalized bias term
        personalized_mu = get_mu(attention_values, self.u_memory, self.base_model, self.device)

        # Get the cross product of the transposed attention values and the user profile
        transposed_att = attention_values.reshape(self.n_k, 1)
        product = torch.mm(transposed_att, p_u)

        # Update the profile memory
        if train:
            self.p_memory = alpha * product + (1 - alpha) * self.p_memory
        self.att_values = attention_values
        return personalized_mu, attention_values

    def write_head(self, u_grads, lr):
        """
        Write head to update the memory
        :param u_grads: gradients collected from training loss
        :param lr: learning rate to control how much new information is kept
        """
        # Update the user-embedding memory matrix
        update_mu(self.att_values, self.u_memory, u_grads, lr)


class TaskMem:
    """
    Create a class to initialize the task-specific memory component
    """
    def __init__(self, n_k, emb_dim, device):
        self.n_k = n_k
        self.memory_UI = torch.rand(n_k, emb_dim * 2, emb_dim * 2, device=device).normal_()
        self.att_values = torch.zeros(n_k)

    def read_head(self, att_values):
        """
        Read head to retrieve the personal preference matrix from the memory cube
        :param att_values: attention values
        """
        self.att_values = att_values
        # Retrieve the user preference matrix
        return get_mui(att_values, self.memory_UI, self.n_k)

    def write_head(self, u_mui, lr):
        """
        Write head to update the memory cube
        :param u_mui: personal preference memory matrix
        :param lr: learning rate that controls how much new preference information is added
        """
        # Update the memory cube
        update_values = update_mui(self.att_values, self.n_k, u_mui)
        self.memory_UI = lr * update_values + (1 - lr) * self.memory_UI


def cosine_similarity(input1, input2):
    """
    Calculate cosine similarity between two data points
    :param input1: first data point
    :param input2: second data point
    :return: The cosine similarity score between the data points
    """
    query_norm = torch.sqrt(torch.sum(input1**2 + 0.00001, 1))
    doc_norm = torch.sqrt(torch.sum(input2**2 + 0.00001, 1))

    prod = torch.sum(torch.mul(input1, input2))
    norm_prod = torch.mul(query_norm, doc_norm)

    cos_sim_raw = torch.div(prod, norm_prod)
    return cos_sim_raw


class Attention(torch.nn.Module):
    """
    Create a class to initialize the attention mechanism
    """
    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, activation_func(activation))
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        # Calculate the cosine similarity between the user profile and the user profile memory
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        # Normalized by softmax function to get the final attention values
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values


def get_mu(att_values, mu, model, device):
    """
    Function to obtain the personalized bias term
    :param att_values: attention values
    :param mu: user embedding memory matrix that stores the memory of the fast gradients
    :param model: model choice
    :param device: device choice
    :return: personalized bias term for feature-specific memory
    """
    mu0, _, _ = model.get_zero_weights()
    attention_values = att_values.reshape(len(mu), 1)

    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu0[j] += attention_values[i] * mu[i][j].to(device)
    return mu0


def update_mu(att_values, mu, grads, lr):
    """
    Function to update the user-embedding memory matrix
    :param att_values: attention values
    :param mu: user embedding memory matrix that stores the memory of the fast gradients
    :param grads: gradients collected from training loss
    :param lr: learning rate to control how much new information is kept
    """
    att_values = att_values.reshape(len(mu), 1)
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu[i][j] = lr * att_values[i] * grads[j] + (1 - lr) * mu[i][j]


def get_mui(att_values, mui, n_k):
    """
    Function to retrieve the user preference matrix
    :param att_values: attention values
    :param mui: memory cube
    :param n_k: number of user preference types
    """
    attention_values = att_values.reshape(n_k, 1, 1)
    attend_mui = torch.mul(attention_values, mui)
    u_mui = attend_mui.sum(dim=0)
    return u_mui


def update_mui(att_values, n_k, u_mui):
    """
    Function to update the user preference matrix after the learning on support set
    :param att_values: attention values
    :param n_k: number of user preference types
    :param u_mui: user preference matrix
    :return: updated user preference matrix
    """
    repeat_u_mui = u_mui.unsqueeze(0).repeat(n_k, 1, 1)
    attention_tensor = att_values.reshape(n_k, 1, 1)
    attend_u_mui = torch.mul(attention_tensor, repeat_u_mui)
    return attend_u_mui
