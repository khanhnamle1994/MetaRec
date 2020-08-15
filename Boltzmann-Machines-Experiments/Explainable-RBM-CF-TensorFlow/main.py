# Import packages
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Weights and Biases
import wandb
wandb.init(entity="khanhnamle1994", project="boltzmann_machines_collaborative_filtering")

# Config is a variable that holds and saves hyper-parameters
config = wandb.config  # Initialize config

# Disable the default activate eager execution in TF v1.0
tf.disable_eager_execution()

# Path directory
path = '../../ml-1m'

# Load data set from MovieLens1M
movies_df = pd.read_csv(path + '/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings_df = pd.read_csv(path + '/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Rename the columns in the dataset
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Create a separate indexing column as we cannot use MovieID
movies_df['List Index'] = movies_df.index

# Merge movies with ratings by MovieID as foreign key
merged_df = movies_df.merge(ratings_df, on='MovieID')

# Drop columns
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

# Get user groups
userGroup = merged_df.groupby('UserID')


def preprocess_data(userGroup, movies_df):
    """
    Format the data for input and output into the RBM
    :param userGroup: data frame that stores user groups
    :param movies_df: data frame that stores movies information
    :return: user ratings normalized into a list - trX
    """
    # No. of users in training
    UsedUsers = 1000
    # create list
    trX = []
    # for each user in the group
    for userID, curUser in userGroup:
        # Temp variable that stores every movie's rating
        temp = [0] * len(movies_df)
        # For Each movie in the curUser's movie list
        for num, movie in curUser.iterrows():
            # Divide ratings by 5 and store it
            temp[movie['List Index']] = movie['Rating'] / 5.0

        # Now add the list of ratings into the training list
        trX.append(temp)
        # Check to see if we finished adding in the amount of users for training
        if UsedUsers == 0:
            break
        UsedUsers -= 1
    return trX


# Return normalized user ratings into a list
trX = preprocess_data(userGroup, movies_df)


def rbm(movies_df, config):
    """
    Implement RBM architecture in TensorFlow
    :param movies_df: data frame that stores movies information
    :param config: variable to store hyper-parameters
    :return: variables to be used during TensorFlow training
    """
    config.n_hid = 100  # Number of hidden layers
    config.n_vis = len(movies_df)  # Number of visible layers

    # Create respective placeholder variables for storing visible and hidden layer biases and weights
    vb = tf.placeholder("float", [config.n_vis])  # Number of unique movies
    hb = tf.placeholder("float", [config.n_hid])  # Number of features
    W = tf.placeholder("float", [config.n_vis, config.n_hid])  # Weights that connect the hidden and visible layers

    # Pre-process the input data
    v0 = tf.placeholder("float", [None, config.n_vis])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

    # Reconstruct the pre-processed input data (Sigmoid and ReLU activation functions are used)
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

    # Set RBM training parameters
    alpha = 0.1  # Set learning rate
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)  # Set positive gradients
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)  # Set negative gradients

    # Calculate contrastive divergence to maximize
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

    # Create methods to update the weights and biases
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    # Set error function (RMSE)
    err = v0 - v1
    err_sum = tf.sqrt(tf.reduce_mean(err**2))

    # Initialize variables
    cur_w = np.zeros([config.n_vis, config.n_hid], np.float32)  # Current weight
    cur_vb = np.zeros([config.n_vis], np.float32)  # Current visible unit biases
    cur_hb = np.zeros([config.n_hid], np.float32)  # Current hidden unit biases
    prv_w = np.zeros([config.n_vis, config.n_hid], np.float32)  # Previous weight
    prv_vb = np.zeros([config.n_vis], np.float32)  # Previous visible unit biases
    prv_hb = np.zeros([config.n_hid], np.float32)  # Previous hidden unit biases

    return v0, W, vb, hb, update_w, prv_w, prv_vb, prv_hb, update_vb, update_hb, cur_w, cur_vb, cur_hb, err_sum


# Return variables from the RBM implementation
v0, W, vb, hb, update_w, prv_w, prv_vb, prv_hb, update_vb, update_hb, cur_w, cur_vb, cur_hb, err_sum = rbm(movies_df, config)

# Initialize TensorFlow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train RBM with 50 epochs and batches of size 512
config.nb_epoch = 50
config.batch_size_ = 512
errors = []

for i in range(config.nb_epoch):
    print("Current epoch: ", i)
    for start, end in zip(range(0, len(trX), config.batch_size_), range(config.batch_size_, len(trX), config.batch_size_)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print("Current RMSE error: ", errors[-1])
    wandb.log({"Train RMSE": errors[-1]})

# Plot errors with respect to number of epochs
plt.plot(errors)
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Number of Epochs')
plt.savefig('pics/result.png')

# We can now predict movies that an arbitrarily selected user might like by feeding in the user's watched
# movie preferences into the RBM and then reconstructing the input

# Selecting the input user
inputUser = [trX[850]]

# Feed in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

# We can then list the 25 most recommended movies for our mock user by sorting it by their scores given by our model
scored_movies_df_850 = movies_df
scored_movies_df_850["Recommendation Score"] = rec[0]
print("\n")
print(scored_movies_df_850.sort_values(["Recommendation Score"], ascending=False).head(25))

# Now we recommend some movies that the user has not yet watched
print("\n")
print(merged_df.iloc[850])

# Now we can find all the movies that our mock user has watched before
movies_df_850 = merged_df[merged_df['UserID'] == 2562]

# We merge all the movies that our mock users has watched with the predicted scores based on his historical data
merged_df_850 = scored_movies_df_850.merge(movies_df_850, on='MovieID', how='outer')
merged_df_850 = merged_df_850.drop('List Index_y', axis=1).drop('UserID', axis=1)
print("\n")
print(merged_df_850.sort_values(["Recommendation Score"], ascending=False).head(25))
