# Import packages
import os
import pandas as pd
from scipy import sparse
import numpy as np


class DataLoader:
    """
    Load Movielens-1M dataset
    """

    def __init__(self):
        """
        Function to initialize the class
        """
        # Ensure that we have the pre-processed data files
        self.pro_dir = os.path.join('processed_data')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"
        # Get number of movies
        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        """
        Function to load processed data
        :param datatype: train, validation, or test sets
        """
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        """
        Function to load the movies
        :return: The number of unique movieIDs
        """
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        """
        Function to load processed train data
        """
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test'):
        """
        Function to load processed validation and test data
        """
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


def get_count(tp, id):
    """
    Function to count triplets
    :param tp: The user, item, rating triplet
    :param id: The indices of the triplets
    :return: The number of triplets
    """
    playcount_groupby_id = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupby_id.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    """
    Function to filter out triplets
    :param tp: The user, item, rating triplet
    :param min_uc: The minimum threshold for user count
    :param min_sc: The minimum threshold for item count
    :return: The filtered triplets, along with the count for users and items
    """
    # Only keep the triplets for items which were clicked on by at least min_sc users
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    """
    Function to split the data into train and test sets
    :param data: Full data
    :param test_prop: Defaulted 20% test set
    :return: The train and test sets
    """
    # Group the data by userId
    data_grouped_by_user = data.groupby('userId')
    # Generate empty list to store train and test data
    tr_list, te_list = list(), list()
    # Generate random seed
    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

    # Concatenate train lists into train data frame
    data_tr = pd.concat(tr_list)
    # Concatenate test lists into test data frame
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    """
    Function to numerize the triplets
    :param tp: The user, item, rating triplet
    :param profile2id: Dictionary of unique user IDs
    :param show2id: Dictionary of unique movie IDs
    :return: Data frame with processed userID and movieID
    """
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':

    print("Load and Preprocess Movielens-1M dataset")

    # location where the MovieLens-1M data sits
    DATA_DIR = '../../ml-1m'

    # Read the ratings data into a data frame
    cols = ['userId', 'movieId', 'rating', 'timestamp']
    dtypes = {'userId': 'int', 'movieId': 'int', 'timestamp': 'int', 'rating': 'int'}
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', names=cols, parse_dates=['timestamp'])

    max_seq_len = 1000

    # Binarize the data (only keep the ratings >= 4)
    ratings = ratings[ratings['rating'] > 3.5]

    # Remove users with greater than $max_seq_len number of watched movies
    ratings = ratings.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)

    # Sort data values with the timestamp
    ratings = ratings.groupby(["userId"]).apply(
        lambda x: x.sort_values(["timestamp"], ascending=True)).reset_index(drop=True)

    # Only keep items that are clicked on by at least 5 users
    ratings, user_activity, item_popularity = filter_triplets(ratings)
    # Calculate the sparsity percentage
    sparsity = 1. * ratings.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (ratings.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    # Get unique id for the user
    unique_uid = user_activity.index
    # Set random seed
    np.random.seed(12345)
    # Shuffle user indices
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # Select 750 users as test users, 750 users as validation users, and the rest of the users for training
    n_users = unique_uid.size
    n_heldout_users = 750

    # Training set
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    # Validation set
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    # Test set
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = ratings.loc[ratings['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    # Path where the processed data will be saved
    pro_dir = os.path.join('processed_data')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    # Get validation data
    vad_plays = ratings.loc[ratings['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
    # Split validation data further into validation-train set and validation-test set
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    # Get test data
    test_plays = ratings.loc[ratings['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
    # Split test data further into test-train set and test-test set
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    # Numerize train data and save to csv format
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    # Numerize train set for validation data and save to csv format
    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)
    # Numerize test set for validation data and save to csv format
    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    # Numerize train set for test data and save to csv format
    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    # Numerize test set for test data and save to csv format
    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")
