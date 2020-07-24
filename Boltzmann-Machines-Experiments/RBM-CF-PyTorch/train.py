# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Import RBM model script
from rbm import RBM

# Path directory
path = '../../ml-1m'

# Load data set from MovieLens1M
movies = pd.read_csv(path + '/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(path + '/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(path + '/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Split the ratings data frame into 75% training and 25% testing
trainingset, testset = train_test_split(ratings, train_size=0.75)
print(len(trainingset), len(testset))

# Convert training and test data into Numpy arrays
trainingset = np.array(trainingset, dtype='int')
testset = np.array(testset, dtype='int')

# Collect the total number of movies and users in order to then make a matrix of the data
nb_users = int(max(max(trainingset[:, 0]), max(testset[:, 0])))
nb_movies = int(max(max(trainingset[:, 1]), max(testset[:, 1])))


# Function to convert the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []  # initialise list
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


# Convert training and test sets into arrays
trainingset = convert(trainingset)
testset = convert(testset)

# Convert training and test sets into torch sensors
training_set = torch.FloatTensor(trainingset)
test_set = torch.FloatTensor(testset)

# Convert ratings (1-5) into binary ratings 1 (liked) and 0 (not liked)
training_set[training_set == 0] = -1  # not rated
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1  # not rated
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Number of movies is the number of visible units
n_vis = len(training_set[0])
# This tunable parameter is the number of features that we want to detect (number of hidden units)
n_hid = 100

# Create the class object RBM()
rbm = RBM(n_vis, n_hid)

batch_size_ = 512  # set batch size to be 512 (tunable)
reconerr = []  # keep track of reconstruction error
nb_epoch = 200  # run for 200 epochs

# Train the RBM
# First for loop - go through every single epoch
for epoch in range(1, nb_epoch + 1):
    train_recon_error = 0  # reconstruction error initialized to 0 at the beginning of training
    s = 0.  # a counter (float type)

    # Second for loop - go through every single user
    # Lower bound is 0, upper bound is (nb_users - batch_size_), batch_size_ is the step of each batch (512)
    # The 1st batch is for user with ID = 0 to user with ID = 511
    for id_user in range(0, nb_users - batch_size_, batch_size_):

        # At the beginning, v0 = vk. Then we update vk
        vk = training_set[id_user:id_user + batch_size_]
        v0 = training_set[id_user:id_user + batch_size_]
        ph0, _ = rbm.sample_h(v0)

        # Third for loop - perform contrastive divergence
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)

            # We don't want to learn when there is no rating by the user, and there is no update when rating = -1
            vk[v0 < 0] = v0[v0 < 0]

        phk, _ = rbm.sample_h(vk)

        # Calculate the loss using contrastive divergence
        rbm.train(v0, vk, ph0, phk)

        # Compare vk updated after the training to v0 (the target)
        train_recon_error += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.

    # Update reconstruction error
    reconerr.append(train_recon_error / s)

    print('Epoch: ' + str(epoch) + '- Reconstruction Error: ' + str(train_recon_error.data.numpy() / s))

# Plot the reconstruction error with respect to increasing number of epochs
plt.plot(reconerr)
plt.ylabel('Training Data Reconstruction Error')
plt.xlabel('Epoch')
plt.show()

# Evaluate the RBM on test set
test_recon_error = 0  # reconstruction error initialized to 0 at the beginning of training
s = 0.  # a counter (float type)

# for loop - go through every single user
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]  # training set inputs are used to activate neurons of my RBM
    vt = test_set[id_user:id_user + 1]  # target

    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

        # Update test reconstruction error
        test_recon_error += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.

print('Reconstruction error:  ' + str(test_recon_error.data.numpy() / s))
