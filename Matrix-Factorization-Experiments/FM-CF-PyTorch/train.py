# Import Python Libraries
import numpy as np
import pandas as pd

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
from FM import *

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

# Number of users, number of items, number of occupations, and number of ranks
n_user = int(fh['n_user'])
n_item = int(fh['n_item'])
n_occu = int(fh['n_occu'])
n_rank = int(fh['n_ranks'])
# Add them up to get number of features
n_feat = n_user + n_item + n_occu + n_rank

train_x[:, 1] += n_user
train_x[:, 2] += n_user + n_item
train_x[:, 3] += n_user + n_item + n_occu
test_x[:, 1] += n_user
test_x[:, 2] += n_user + n_item
test_x[:, 3] += n_user + n_item + n_occu

# Hyperparameters
lr = 1e-2 # learning rate
k = 10 # number of dimensions per user and item
# New parameter for regularizing bias and features
c_bias = 1e-6
c_feat = 1e-6

# Setup logging
log_dir = 'runs/simple_mf_06_fm_' + str(datetime.now()).replace(' ', '_')
writer = SummaryWriter(log_dir=log_dir)

# Instantiate the model class object
model = MF(n_feat, k=k, c_bias=c_bias, c_feat=c_feat, writer=writer)
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
    evaluat.run(test_loader) # When triggered, run the validation set
    metrics = evaluat.state.metrics # Keep track of metrics
    avg_accuracy = metrics['accuracy']
    print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_accuracy))
    writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Run the model for 50 epochs
trainer.run(train_loader, max_epochs=50)
