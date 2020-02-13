# Import NumPy and PyTorch
import numpy as np
import torch

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
from MF import *

# Load preprocessed data
path = '../../ml-1m/'
full_data = np.load(path + 'dataset.npz')

# We have a bunch of feature columns and last column is the y-target
# Note that Pytorch is finicky about need int64 types
train_x = full_data['train_x'].astype(np.int64)
train_y = full_data['train_y']

# We've already split the data into train & test set
test_x = full_data['test_x'].astype(np.int64)
test_y = full_data['test_y']

# Extract the number of users and number of items
n_user = int(full_data['n_user'])
n_item = int(full_data['n_item'])

# Define the Hyper-parameters
lr = 1e-2  # Learning Rate
k = 10  # Number of dimensions per user, item
c_vector = 1e-6  # regularization constant

# Setup TensorBoard logging
log_dir = 'runs/simple_mf_01_' + str(datetime.now()).replace(' ', '_')
writer = SummaryWriter(log_dir=log_dir)

# Instantiate the MF class object
model = MF(n_user, n_item, writer=writer, k=k, c_vector=c_vector)

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create a supervised trainer
trainer = create_supervised_trainer(model, optimizer, model.loss)

# Use Mean Squared Error as evaluation metric
metrics = {'evaluation': MeanSquaredError()}

# Create a supervised evaluator
evaluator = create_supervised_evaluator(model, metrics=metrics)

# Load the train and test data
train_loader = Loader(train_x, train_y, batchsize=1024)
test_loader = Loader(test_x, test_y, batchsize=1024)


def log_training_loss(engine, log_interval=500):
    """
    Function to log the training loss
    """
    model.itr = engine.state.iteration  # Keep track of iterations
    if model.itr % log_interval == 0:
        fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        # Keep track of epochs and outputs
        msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
        print(msg)


trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)


def log_validation_results(engine):
    """
    Function to log the validation loss
    """
    # When triggered, run the validation set
    evaluator.run(test_loader)
    # Keep track of the evaluation metrics
    avg_loss = evaluator.state.metrics['evaluation']
    print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_loss))
    writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)


trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Run the model for 50 epochs
trainer.run(train_loader, max_epochs=50)
