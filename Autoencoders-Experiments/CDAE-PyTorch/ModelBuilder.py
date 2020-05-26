# Import models
from DAE import DAE
from CDAE import CDAE


def build_model(model_name, model_conf, num_users, num_items, device):
    if model_name == 'DAE':
        model = DAE(model_conf, num_users, num_items, device)
    elif model_name == 'CDAE':
        model = CDAE(model_conf, num_users, num_items, device)
    else:
        raise NotImplementedError('Choose correct model name.')

    return model
