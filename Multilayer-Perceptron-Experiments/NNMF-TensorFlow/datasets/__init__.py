from .ml_1m import load_ml_1m_data
from . import utils
from .utils import ML_1M


def load_data(kind):
    utils.make_dir_if_not_exists('data')

    if kind == ML_1M:
        return load_ml_1m_data()
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))


def get_N_and_M(kind):
    if kind == ML_1M:
        return 6040, 3952
    else:
        raise NotImplementedError(
            "Kind '{}' is not implemented yet.".format(kind))