# Import packages
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(data_path):
    """
    Load data
    :param data_path: path to dataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_matrix, test_matrix, user_id_map, user_popularity, item_id_map, item_popularity, num_uesrs, num_items = \
        data['train_mat'], data['test_mat'], data['user_id_dict'], data['user_popularity'], data['item_id_dict'], data[
            'item_popularity'], data['num_users'], data['num_items']

    return train_matrix, test_matrix, user_id_map, user_popularity, item_id_map, item_popularity, num_uesrs, num_items


def preprocess(data_path, save_path, stat_path, sep, train_ratio=0.8, binarize_threshold=0.0, order_by_popularity=True):
    """
    [1]
    Read raw data.
    Binarize ratings into implicit feedbacks.
    Add old user/item id into new consecutive ids.
    """
    print('Preprocess starts.')
    print("Loading the dataset from \"%s\"" % data_path)
    # Read data into Pandas dataframe
    data = pd.read_csv(data_path, sep=sep, names=['user', 'item', 'ratings', 'timestamps'],
                       dtype={'user': int, 'item': int, 'ratings': float, 'timestamps': float},
                       engine='python')

    # initial # user, items
    num_users = len(pd.unique(data.user))
    num_items = len(pd.unique(data.item))

    print('initial user, item:', num_users, num_items)

    if binarize_threshold > 0.0:
        print("Binarize ratings greater than or equal to %.f" % binarize_threshold)
        data = data[data['ratings'] >= binarize_threshold]

    # convert ratings into implicit feedback
    data['ratings'] = 1.0

    num_items_by_user = data.groupby('user', as_index=False).size()
    num_users_by_item = data.groupby('item', as_index=False).size()

    # assign new user id
    print('Assign new user id...')
    user_frame = num_items_by_user.to_frame()
    user_frame.columns = ['item_cnt']

    if order_by_popularity:
        user_frame = user_frame.sort_values(by='item_cnt', ascending=False)
    user_frame['new_id'] = list(range(num_users))

    frame_dict = user_frame.to_dict()
    user_id_dict = frame_dict['new_id']
    user_frame = user_frame.set_index('new_id')
    user_to_num_items = user_frame.to_dict()['item_cnt']

    data.user = [user_id_dict[x] for x in data.user.tolist()]

    # assign new item id
    print('Assign new item id...')
    item_frame = num_users_by_item.to_frame()
    item_frame.columns = ['user_cnt']
    if order_by_popularity:
        item_frame = item_frame.sort_values(by='user_cnt', ascending=False)
    item_frame['new_id'] = range(num_items)

    frame_dict = item_frame.to_dict()
    item_id_dict = frame_dict['new_id']
    item_frame = item_frame.set_index('new_id')
    item_to_num_users = item_frame.to_dict()['user_cnt']
    data.item = [item_id_dict[x] for x in data.item.tolist()]

    num_users, num_items = len(user_id_dict), len(item_id_dict)
    num_ratings = len(data)

    """
    [2]
    Preprocess UIRT raw data into trainable form.
    Holdout feedbacks for test per user.
    Save preprocessed data.
    """
    # Split data into train/test
    print('Split data into train/test.')
    data_group = data.groupby('user')
    train_list, test_list = [], []

    num_zero_train, num_zero_test = 0, 0
    for _, group in data_group:
        user = pd.unique(group.user)[0]
        num_items_user = len(group)
        num_train = int(train_ratio * num_items_user)
        num_test = num_items_user - num_train

        group = group.sort_values(by='timestamps')

        idx = np.ones(num_items_user, dtype='bool')

        test_idx = np.random.choice(num_items_user, num_test, replace=False)
        idx[test_idx] = False

        if len(group[idx]) == 0:
            num_zero_train += 1
        else:
            train_list.append(group[idx])

        if len(group[np.logical_not(idx)]) == 0:
            num_zero_test += 1
        else:
            test_list.append(group[np.logical_not(idx)])

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    print('# zero train, test: %d, %d' % (num_zero_train, num_zero_test))

    train_sparse = df_to_sparse(train_df, shape=(num_users, num_items))
    test_sparse = df_to_sparse(test_df, shape=(num_users, num_items))

    # Save data and statistics
    data_to_save = {
        'train_mat': train_sparse,
        'test_mat': test_sparse,
        'user_id_dict': user_id_dict,
        'user_popularity': user_to_num_items,
        'item_id_dict': item_id_dict,
        'item_popularity': item_to_num_users,
        'num_users': num_users,
        'num_items': num_items
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    ratings_per_user = list(user_to_num_items.values())

    info_lines = []
    info_lines.append('# users: %d, # items: %d, # ratings: %d' % (num_users, num_items, num_ratings))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    info_lines.append("Min/Max/Avg. ratings per users (full data): %d %d %.2f\n" % (
    min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))

    info_lines.append('# train users: %d, # train ratings: %d' % (train_sparse.shape[0], train_sparse.nnz))
    info_lines.append('# test users: %d, # test ratings: %d' % (test_sparse.shape[0], test_sparse.nnz))

    with open(stat_path, 'wt') as f:
        f.write('\n'.join(info_lines))

    print('Preprocess finished.')


def df_to_sparse(df, shape):
    rows, cols = df.user, df.item
    values = df.ratings

    sp_data = sp.csr_matrix((values, (rows, cols)), dtype='float64', shape=shape)

    num_nonzeros = np.diff(sp_data.indptr)
    rows_to_drop = num_nonzeros == 0
    if sum(rows_to_drop) > 0:
        print('%d empty users are dropped from matrix.' % sum(rows_to_drop))
        sp_data = sp_data[num_nonzeros != 0]

    return sp_data