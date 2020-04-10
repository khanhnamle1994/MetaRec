from .ml_1m import load_ml_1m_data
from . import utils
from .utils import ML_1M


def load_data(kind):
    """
        Function to load data
    """
    utils.make_dir_if_not_exists('data')

    if kind == ML_1M:
        return load_ml_1m_data()
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))


def get_N_and_M(kind):
    """
        Function to get number of users and number of items
    """
    if kind == ML_1M:
        return 6040, 3952  # 6040 movies and 3952 movies
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))
