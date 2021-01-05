# Import libraries
import hashlib
import os
import pickle
import random
import numpy as np
import torch


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)), and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    """
    Save an object
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load an object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_path_from_args(args):
    """
    Returns a unique hash for an argparse object
    """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path


def get_base_path():
    """
    Get path directory
    """
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I do not know where I am; please specify a path for saving results.')
