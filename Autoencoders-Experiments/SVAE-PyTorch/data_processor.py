import os
import pandas as pd
import numpy as np


def get_count(tp, id):
    """
    Function to count triplets
    :param tp: The user, item, rating triplet
    :param id: The indices of the triplets
    :return: The number of triplets
    """
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    """
    Function to filter out triplets
    :param tp: The user, item, rating triplet
    :param min_uc: The minimum threshold for user count
    :param min_sc: The minimum threshold for item count
    :return: The filtered triplets, along with the count for users and items
    """
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
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

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            # idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            idx[int((1.0 - test_prop) * n_items_u):] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    # Concatenate train lists into train data frame
    data_tr = pd.concat(tr_list)
    # Concatenate test lists into test data frame
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp):
    """
    Function to numerize the triplets
    :param tp: The user, item, rating triplet
    :return: Data frame with processed userID, itemID, and rating
    """
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    ra = list(map(lambda x: x, tp['rating']))
    ret = pd.DataFrame(data={'uid': uid, 'sid': sid, 'rating': ra}, columns=['uid', 'sid', 'rating'])
    ret['rating'] = ret['rating'].apply(pd.to_numeric)
    return ret


if __name__ == '__main__':

    print("Load and Preprocess Movielens-1M dataset")

    # Location where the MovieLens-1M data sits
    DATA_DIR = '../../ml-1m'

    # Path where the processed data will be saved
    processed_dir = os.path.join('processed_data')

    # We don't want to keep pre-processing every time we run the code
    if not os.path.isdir(processed_dir):
        cols = ['userId', 'movieId', 'rating', 'timestamp']
        dtypes = {'userId': 'int', 'movieId': 'int', 'timestamp': 'int', 'rating': 'int'}
        raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', names=cols, parse_dates=['timestamp'])

        max_seq_len = 1000

        # Binarize the data (only keep ratings >= 4)
        raw_data = raw_data[raw_data['rating'] > 3.5]

        # Remove users with greater than $max_seq_len number of watched movies
        raw_data = raw_data.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)

        # Sort data values with the timestamp
        raw_data = raw_data.groupby(["userId"]).apply(
            lambda x: x.sort_values(["timestamp"], ascending=True)).reset_index(drop=True)

        # Only keep items that are clicked on by at least 5 users
        raw_data, user_activity, item_popularity = filter_triplets(raw_data)
        # Calculate the sparsity percentage
        sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
        print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
              (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

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

        train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
        unique_sid = pd.unique(train_plays['movieId'])

        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        with open(os.path.join(processed_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)

        # Get validation data
        vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
        vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
        # Split validation data further into validation-train set and validation-test set
        vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

        # Get test data
        test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
        test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
        # Split test data further into test-train set and test-test set
        test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

        # Numerize train data and save to csv format
        train_data = numerize(train_plays)
        train_data.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
        # Numerize train set for validation data and save to csv format
        vad_data_tr = numerize(vad_plays_tr)
        vad_data_tr.to_csv(os.path.join(processed_dir, 'validation_tr.csv'), index=False)
        # Numerize test set for validation data and save to csv format
        vad_data_te = numerize(vad_plays_te)
        vad_data_te.to_csv(os.path.join(processed_dir, 'validation_te.csv'), index=False)
        # Numerize train set for test data and save to csv format
        test_data_tr = numerize(test_plays_tr)
        test_data_tr.to_csv(os.path.join(processed_dir, 'test_tr.csv'), index=False)
        # Numerize test set for test data and save to csv format
        test_data_te = numerize(test_plays_te)
        test_data_te.to_csv(os.path.join(processed_dir, 'test_te.csv'), index=False)

        print("Done!")