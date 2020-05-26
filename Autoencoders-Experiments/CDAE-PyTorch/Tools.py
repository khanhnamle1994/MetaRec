import torch.nn.functional as F


def apply_activation(act_name, x):
    if act_name == 'sigmoid':
        return F.sigmoid(x)
    elif act_name == 'tanh':
        return F.tanh(x)
    elif act_name == 'relu':
        return F.relu(x)
    elif act_name == 'elu':
        return F.elu(x)
    else:
        raise NotImplementedError('Choose appropriate activation function. (current input: %s)' % act_name)


class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.total = 0

    def update(self, value):
        self.sum += value
        self.total += 1

    @property
    def mean(self):
        return self.sum / self.total