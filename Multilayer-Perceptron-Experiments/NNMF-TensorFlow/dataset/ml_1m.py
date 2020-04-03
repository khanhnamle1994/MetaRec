import pandas as pd

from . import utils
from .utils import ML_1M as KEY

_COL_NAMES = ['user_id', 'item_id', 'rating', 'timestamp']
_DELIMITER = '::'


def load_ml_1m_data():
    utils.download_data_if_not_exists(
        KEY, 'http://files.grouplens.org/datasets/movielens/ml-1m.zip')

    base_file_name, test_file_name = utils.split_data(KEY, 'ratings.dat',
                                                      ('base', 'test'))
    train_file_name, valid_file_name = utils.split_data(
        KEY, base_file_name, ('train', 'valid'))

    train_data = pd.read_csv(
        utils.get_file_path(KEY, train_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    train_data['user_id'] = train_data['user_id'] - 1
    train_data['item_id'] = train_data['item_id'] - 1

    valid_data = pd.read_csv(
        utils.get_file_path(KEY, valid_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    valid_data['user_id'] = valid_data['user_id'] - 1
    valid_data['item_id'] = valid_data['item_id'] - 1

    test_data = pd.read_csv(
        utils.get_file_path(KEY, test_file_name),
        delimiter=_DELIMITER,
        header=None,
        names=_COL_NAMES)
    test_data['user_id'] = test_data['user_id'] - 1
    test_data['item_id'] = test_data['item_id'] - 1

    return {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
    }