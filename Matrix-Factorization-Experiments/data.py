# Import Libraries
import numpy as np
import pandas as pd

# Set the directory path to MovieLens 1M dataset
path = '../ml-1m/'

'''
Create a Pandas Data Frame for the "Users" csv file
The column names are: user, gender, age, occupation, and zipcode
'''
users = pd.read_csv(path + "users.dat", delimiter='::', engine='python',
                    names=['user', 'gender', 'age', 'occupation', 'zipcode'])

'''
Create a Pandas Data Frame for the "Ratings" csv file
The column names are: user, item, rating, timestamp
'''
rat = pd.read_csv(path + "ratings.dat", delimiter='::', engine='python',
                  names=['user', 'item', 'rating', 'timestamp'])

# Is this rating the first rating ever for that user, or the nth?
rat['rank'] = rat.groupby("user")["timestamp"].rank(ascending=True)

# Set a random seed to make our numbers predictable
np.random.seed(42)

# Split the ratings data into 75% training set and 25% test set
rat['is_train'] = np.random.random(len(rat)) < 0.75
rat.to_pickle(path + "dataset.pd")

# Merge ratings & user features into one Data Frame via the 'user' column
df = rat.merge(users, on='user')

# Compute cardinality
n_features = df.user.max() + 1 + df.item.max() + 1
n_user = df.user.max() + 1
n_item = df.item.max() + 1
n_rank = df['rank'].max() + 1
n_occu = df['occupation'].max() + 1

# Verify the number of items, number of users, number of features, number of occupations, and number of rows
print('Number of Items:', n_item)
print('Number of Users:', n_user)
print('Number of Features:', n_features)
print('Number of Occupations:', n_occu)
print('Number of Rows:', len(df))


# Function to split data to features and target
def split(subset):
    # The features include 'user', 'item', 'rank', and 'occupation
    feat_cols = ['user', 'item', 'rank', 'occupation']
    features = subset[feat_cols]
    features = features.values.astype(np.int32)

    # The target is 'rating'
    target_cols = ['rating']
    target = subset[target_cols]
    target = target.values.astype(np.float32)

    # Retain a NumPy array of both the features and target
    both = subset[feat_cols + target_cols]
    return features, target, both


# Apply "split" function to both the training set and test set
train_x, train_y, train_xy = split(df[df.is_train])
test_x, test_y, test_xy = split(df[~df.is_train])

# Save this into a file called "dataset.npz"
np.savez(path + "dataset.npz",
         train_x=train_x, train_y=train_y, train_xy=train_xy,
         test_x=test_x, test_y=test_y, test_xy=test_xy,
         n_user=n_user, n_item=n_item, n_ranks=n_rank, n_occu=n_occu)
