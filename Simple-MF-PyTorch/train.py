# Import Packages
import os.path
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile
from SimpleMF import *

# Load preprocessed data
path = '/Users/khanhnamle/Desktop/CSCI799-Graduate-Independent-Study/Codebase/ml-1m/'
fh = np.load(path + 'dataset.npz')

# We have a bunch of feature columns and last column is the y-target
# Note pytorch is finicky about need int64 types
train_x = fh['train_x'].astype(np.int64)
train_y = fh['train_y']

# We've already split into train & test
test_x = fh['test_x'].astype(np.int64)
test_y = fh['test_y']

# Number of users and number of items
n_user = int(fh['n_user'])
n_item = int(fh['n_item'])
