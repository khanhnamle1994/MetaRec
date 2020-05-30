# Import model
from ESAE import ESAE


def build_model(model_name, model_conf, num_users, num_items, device):
    """
    Function to build the model
    :param model_name: choice of model
    :param model_conf: model configuration
    :param num_users: number of users
    :param num_items: number of items
    :param device: choice of device
    :return: the model
    """
    if model_name == 'ESAE':
        model = ESAE(model_conf, num_users, num_items, device)
    else:
        raise NotImplementedError('Choose correct model name.')

    return model
