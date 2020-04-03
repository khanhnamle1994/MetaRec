import os
import random

ML_1M = 'ml-1m'


def get_file_path(key, file_name):
    return os.path.join('data/{}/{}'.format(key, file_name))


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download_data_if_not_exists(kind, url):
    if not os.path.exists('data/{}'.format(kind)):
        os.system('wget {url} -O data/{kind}.zip'.format(url=url, kind=kind))
        os.system('unzip data/{}.zip -d data'.format(kind))


def _is_file_exists(key, file_name):
    file_path = get_file_path(key, file_name)
    return os.path.exists(file_path)


def _is_files_exists(key, file_names):
    if len(file_names) == 0:
        return True
    if _is_file_exists(key, file_names[0]):
        return _is_file_exists(key, file_names[1:])


def _get_file_names_from_post_fixes(file_name, post_fixes):
    return [file_name + '.' + post_fix for post_fix in post_fixes]


def _is_splitted(key, file_name, post_fixes):
    return _is_files_exists(key,
                            _get_file_names_from_post_fixes(
                                file_name, post_fixes))


def split_data(key, file_name, post_fixes, rate=0.9):
    assert len(post_fixes) == 2

    a_file_name, b_file_name = _get_file_names_from_post_fixes(
        file_name, post_fixes)
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