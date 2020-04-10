import os
import random

ML_1M = 'ml-1m'


def get_file_path(key, file_name):
    """
        Function to get data path
    """
    return os.path.join('data/{}/{}'.format(key, file_name))


def make_dir_if_not_exists(path):
    """
        Function to make a new directory path if the current path does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)


def download_data_if_not_exists(kind, url):
    """
        Function to download data with `wget` and `unzip`
    """
    if not os.path.exists('data/{}'.format(kind)):
        os.system('wget {url} -O data/{kind}.zip'.format(url=url, kind=kind))
        os.system('unzip data/{}.zip -d data'.format(kind))


def _is_file_exists(key, file_name):
    """
        Function to check if a file exists
    """
    file_path = get_file_path(key, file_name)
    return os.path.exists(file_path)


def _is_files_exists(key, file_names):
    """
        Function to check if multiple files exist
    """
    if len(file_names) == 0:
        return True
    if _is_file_exists(key, file_names[0]):
        return _is_file_exists(key, file_names[1:])


def _get_file_names_from_post_fixes(file_name, post_fixes):
    """
        Function to get the file names
    """
    return [file_name + '.' + post_fix for post_fix in post_fixes]


def _is_splitted(key, file_name, post_fixes):
    """
        Function to check if the data is already splitted
    """
    return _is_files_exists(key,
                            _get_file_names_from_post_fixes(
                                file_name, post_fixes))


def split_data(key, file_name, post_fixes, rate=0.9):
    """
        Function to split the data into 90 / 10 ratio
    """
    assert len(post_fixes) == 2

    a_file_name, b_file_name = _get_file_names_from_post_fixes(file_name, post_fixes)
    if not _is_splitted(key, file_name, post_fixes):
        file_path = get_file_path(key, file_name)
        a_file_path = get_file_path(key, a_file_name)
        b_file_path = get_file_path(key, b_file_name)

        with open(file_path, 'r') as f, \
             open(a_file_path, 'w') as a_f, \
             open(b_file_path, 'w') as b_f:
            for line in f:
                if random.random() < rate:
                    a_f.write(line)
                else:
                    b_f.write(line)

    return a_file_name, b_file_name