# Import library
import torch

# Import utility script
from utils import activation_func


class RecMAM(torch.nn.Module):
    """
    Create a class to initialize the recommendation model
    """
    def __init__(self, embedding_dim, n_y, n_layer, activation='sigmoid', classification=True):
        """
        Initialize the class
        :param embedding_dim: the embedding dimension
        :param n_y: number of labels
        :param n_layer: number of layers
        :param activation: choice of activation function
        :param classification: Boolean value to turn on classification mode
        """
        super(RecMAM, self).__init__()
        self.input_size = embedding_dim * 2

        self.mem_layer = torch.nn.Linear(self.input_size, self.input_size)

        fcs = []
        last_size = self.input_size

        for i in range(n_layer - 1):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fcs.append(linear_model)
            last_size = out_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        if classification:
            finals = [torch.nn.Linear(last_size, n_y), activation_func('softmax')]
        else:
            finals = [torch.nn.Linear(last_size, 1)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x1, x2):
        """
        Perform a forward pass
        :param x1: Input user info
        :param x2: Input item info
        :return: Output
        """
        x = torch.cat([x1, x2], 1)
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)
        return out
