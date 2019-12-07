import os.path
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile

path = '/Users/khanhnamle/Desktop/CSCI799-Graduate-Independent-Study/Codebase/ml-1m/'

# Create users dataframe
users = pd.read_csv(path + "users.dat", delimiter='::', engine='python', names=['user', 'gender', 'age', 'occupation', 'zipcode'])

# Create ratings dataframe
rat = pd.read_csv(path + "ratings.dat", delimiter='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')

# Is this rating the first rating ever for that user, or the nth?
rat['rank'] = rat.groupby("user")["timestamp"].rank(ascending=True)

# Make our numbers predictable
np.random.seed(42)

# Set 75% of dataset to training and 25% test
rat['is_train'] = np.random.random(len(rat)) < 0.75
rat.to_pickle(path + "dataset.pd")

# Merge ratings & user features
df = rat.merge(users, on='user')

# Compute cardinalities
n_features = df.user.max() + 1 + df.item.max() + 1
n_user = df.user.max() + 1
n_item = df.item.max() + 1
n_rank = df['rank'].max() + 1
n_occu = df['occupation'].max() + 1
print('n_item', n_item)
print('n_user', n_user)
print('n_featuers', n_features)
print('n_occu', n_occu)
print('n_rows', len(df))

# Function to split data to training and test sets
def split(subset):
    feat_cols = ['user', 'item', 'rank', 'occupation']
    out_cols = ['rating']
    features = subset[feat_cols]
    outcomes = subset[out_cols]
    features = features.values.astype(np.int32)
    outcomes = outcomes.values.astype(np.float32)
    both = subset[feat_cols + out_cols]
    return features, outcomes, both

# Apply "split" function
train_x, train_y, train_xy = split(df[df.is_train])
test_x, test_y, test_xy = split(df[~df.is_train])

np.savez(path + "dataset.npz",
        train_x=train_x, train_y=train_y, train_xy=train_xy,
        test_x=test_x, test_y=test_y, test_xy=test_xy,
        n_user=n_user, n_item=n_item, n_ranks=n_rank, n_occu=n_occu)
