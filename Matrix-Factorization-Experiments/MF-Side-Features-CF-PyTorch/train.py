# Import Python Libraries
import os.path
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile

# Import PyTorch Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics import MeanSquaredError

# Import Tensorboard
from tensorboardX import SummaryWriter

# Import Utility Functions
from loader import Loader
from datetime import datetime

# Import the Model Script
from MFSideFeat import *

# Load preprocessed data
path = '../../ml-1m/'
fh = np.load(path + 'dataset.npz')

# We have a bunch of feature columns and last column is the y-target
# Note pytorch is finicky about need int64 types
train_x = fh['train_x'].astype(np.int64)
train_y = fh['train_y']

# We've already split into train & test
test_x = fh['test_x'].astype(np.int64)
test_y = fh['test_y']

# Number of users, number of items and number of occupations
n_user = int(fh['n_user'])
n_item = int(fh['n_item'])
n_occu = int(fh['n_occu'])

# Hyperparameters
lr = 1e-2
# Number of dimensions per user, item
k = 10
# New parameter for regularizing bias and side features
c_bias = 1e-6
c_vector = 1e-6

# Setup logging
log_dir = 'runs/simple_mf_03_side_features_' + str(datetime.now()).replace(' ', '_')
writer = SummaryWriter(log_dir=log_dir)

# Instantiate the model class object
model = MF(n_user, n_item, n_occu, writer=writer, k=k, c_bias=c_bias, c_vector=c_vector)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create a supervised trainer
trainer = create_supervised_trainer(model, optimizer, model.loss)
# Use Mean Squared Error as accuracy metric
metrics = {'accuracy': MeanSquaredError()}
# Create a supervised evaluator
evaluat = create_supervised_evaluator(model, metrics=metrics)

# Load the train and test datasets
train_loader = Loader(train_x, train_y, batchsize=1024)
test_loader = Loader(test_x, test_y, batchsize=1024)

def log_training_loss(engine, log_interval=500):
    '''
    Function to log the training loss
    '''
    epoch = engine.state.epoch # Keep track of epochs
    itr = engine.state.iteration # Keep track of iterations
    fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
    msg = fmt.format(epoch, itr, len(train_loader), engine.state.output) # Keep track of outputs
    model.itr = itr
    if itr % log_interval == 0:
        print(msg)

trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)

def log_validation_results(engine):
    '''
    Function to log the validation results
    '''
    # When triggered, run the validation set
    evaluat.run(test_loader)
    metrics = evaluat.state.metrics # Keep track of metrics
    avg_accuracy = metrics['accuracy']
    print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_accuracy))
    writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Run the model for 40 epochs
trainer.run(train_loader, max_epochs=40)
