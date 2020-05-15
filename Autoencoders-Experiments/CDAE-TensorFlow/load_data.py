import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_data_all():
    """
    Function to load the MovieLens 1M ratings excluding the negative ratings
    """
    # Read ratings file into a data-frame
    path = "../../ml-1m/ratings.dat"
    df = pd.read_csv(path, delimiter='::', names=['user_id', 'item_id', 'rating', 'time'], engine='python')

    # Get the number of users and number of items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    # Get the train and test set (80/20 split ratio defaulted)
    train_data, test_data = train_test_split(df, test_size=0.2)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # Empty arrays to store the rows, columns, and targets from train set
    train_row = []
    train_col = []
    train_rating = []

    # Empty dictionary to store (user, item) pairs in train set
    train_dict = {}
    # In train_dict, the keys are (user, item) pairs and the values are their counts
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1
    # Populate train_row, train_col, and train_rating
    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                train_rating.append(1)
            else:
                train_rating.append(0)

    # Use a Compressed Sparse Row matrix to represent the train set
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    # Empty dictionary to store the items not yet been rated by a specific user
    neg_items = {}
    # Empty array to store the user-item interaction matrix in train set
    train_interaction_matrix = []
    # Populate neg_items and train_interaction_matrix
    for u in range(n_users):
        neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    # Empty arrays to store the rows, columns, and targets from test set
    test_row = []
    test_col = []
    test_rating = []
    # Populate test_row, test_col, and test_rating
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)

    # Use a Compressed Sparse Row matrix to represent the test set
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    # Empty dictionary to store test set
    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_interaction_matrix, test_dict, n_users, n_items


def load_data_neg():
    """
    Function to load all the MovieLens 1M ratings including the negative ratings
    """
    # Read ratings file into a data-frame
    path = "../../ml-1m/ratings.dat"
    df = pd.read_csv(path, delimiter='::', names=['user_id', 'item_id', 'rating', 'time'], engine='python')

    # Get the number of users and number of items
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    # Get the train and test set (80/20 split ratio defaulted)
    train_data, test_data = train_test_split(df, test_size=0.2)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # Empty arrays to store the rows, columns, and targets from train set
    train_row = []
    train_col = []
    train_rating = []
    # Populate train_row, train_col, and train_rating
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    # Use a Compressed Sparse Row matrix to represent the train set
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # Empty arrays to store the rows, columns, and targets from test set
    test_row = []
    test_col = []
    test_rating = []
    # Populate test_row, test_col, and test_rating
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    # Use a Compressed Sparse Row matrix to represent the test set
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    # Empty dictionary to store test set
    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_matrix.todok(), test_dict, n_users, n_items
