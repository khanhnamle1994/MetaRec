# CometML for experiment logging
from comet_ml import Experiment

# Import packages
import os
import torch
import datetime as dt
import time
import matplotlib.pyplot as plt

# Import utility scripts
from data_parser import load_data
from utils import file_write
from model import Model, VAELoss
from train import train
from evaluate import evaluate
from utils import plot_len_vs_ndcg

# Create an experiment with API key
experiment = Experiment(api_key="GaicgDHvizDRCbpq2wVV8NHnX", project_name="autoencoders-movielens1M")

# Dictionary to store model hyper-parameters
hyper_params = {
    'project_name': 'svae_ml1m',  # Project name
    'model_file_name': '',  # Model file name
    'log_file': '',  # Log file name
    'history_split_test': [0.8, 0.2],  # Part of test history to train on : Part of test history to test
    'learning_rate': 0.01,  # Choice of learning rate (required only if optimizer is AdaGrad)
    'optimizer': 'adam',  # Choice of optimizer defaulted to Adam
    'weight_decay': float(5e-3),  # Choice of weight decay defaulted to 0.005
    'epochs': 50,  # Number of epochs
    'batch_size': 1,  # Needs to be 1, because we don't pack multiple sequences in the same batch
    'item_embed_size': 256,  # Item embedding layer of size 256
    'rnn_size': 200,  # Recurrent layer realized as a GRU with 200 cells
    'hidden_size': 150,  # Encoding layer of size 150
    'latent_size': 64,  # Number of latent factors set to 64
    'loss_type': 'next_k',  # [predict_next, same, prefix, postfix, exp_decay, next_k]
    'next_k': 4,  # Size for the number of items forward in time to predict on
    'number_users_to_keep': 1000000000,  # Number of held-out users for evaluation purpose
    'batch_log_interval': 1000,  # Log metrics after this number of batches
    'train_cp_users': 200,
    'exploding_clip': 0.25,  # Exploding gradient clipping
}

# Log model hyper-parameters
experiment.log_parameters(hyper_params)

# Store the optimizer, weight decay, loss type, item embedding size, RNN size, and latent dimension size
file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] == 'adagrad':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_weight_decay_' + str(hyper_params['weight_decay'])
file_name += '_loss_type_' + str(hyper_params['loss_type'])
file_name += '_item_embed_size_' + str(hyper_params['item_embed_size'])
file_name += '_rnn_size_' + str(hyper_params['rnn_size'])
file_name += '_latent_size_' + str(hyper_params['latent_size'])

# Path to store the log file and the model file
log_file_root = "saved_logs/"
model_file_root = "saved_models/"
if not os.path.isdir(log_file_root):
    os.mkdir(log_file_root)
if not os.path.isdir(model_file_root):
    os.mkdir(model_file_root)
hyper_params['log_file'] = log_file_root + hyper_params['project_name'] + '_log' + file_name + '.txt'
hyper_params['model_file_name'] = model_file_root + hyper_params['project_name'] + '_model' + file_name + '.pt'

# Load the processed data and get the reader classes for training, test, and validation sets
train_reader, val_reader, test_reader, total_items = load_data(hyper_params)
hyper_params['total_items'] = total_items
hyper_params['testing_batch_limit'] = test_reader.num_b

file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
file_write(hyper_params['log_file'], "Data reading complete!")
file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(train_reader.num_b))
file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(val_reader.num_b))
file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(test_reader.num_b))
file_write(hyper_params['log_file'], "Total Items: " + str(total_items) + "\n")

# Instantiate the model
model = Model(hyper_params)
# Loss function is the VAE loss
criterion = VAELoss(hyper_params)

# Different options for the optimizer
if hyper_params['optimizer'] == 'adagrad':
    # AdaGrad
    optimizer = torch.optim.Adagrad(
        model.parameters(), weight_decay=hyper_params['weight_decay'], lr=hyper_params['learning_rate']
    )
elif hyper_params['optimizer'] == 'adadelta':
    # AdaDelta
    optimizer = torch.optim.Adadelta(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )
elif hyper_params['optimizer'] == 'adam':
    # Adam
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )
elif hyper_params['optimizer'] == 'rmsprop':
    # RMSProp
    optimizer = torch.optim.RMSprop(
        model.parameters(), weight_decay=hyper_params['weight_decay']
    )

file_write(hyper_params['log_file'], str(model))
file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

best_val_ndcg = None

try:
    for epoch in range(1, hyper_params['epochs'] + 1):
        # Keep track the time for every epoch
        epoch_start_time = time.time()

        # Perform training
        train(model, criterion, train_reader, optimizer, epoch, hyper_params, experiment)

        # Calculate the metrics on the train set
        metrics, _ = evaluate(model, criterion, train_reader, hyper_params, True, experiment)

        string = ""
        for m in metrics:
            string += " | " + m + ' = ' + str(metrics[m])
        string += ' (TRAIN)'

        # Calculate the metrics on the validation set
        metrics, _ = evaluate(model, criterion, val_reader, hyper_params, False, experiment)

        string2 = ""
        for m in metrics:
            string2 += " | " + m + ' = ' + str(metrics[m])
        string2 += ' (VAL)'

        ss = '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string
        ss += '\n'
        ss += '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string2
        ss += '\n'
        ss += '-' * 89
        file_write(hyper_params['log_file'], ss)

        # Log the best NDCG@100 metric on the validation set
        if not best_val_ndcg or metrics['NDCG@100'] >= best_val_ndcg:
            with open(hyper_params['model_file_name'], 'wb') as f:
                torch.save(model, f)
            best_val_ndcg = metrics['NDCG@100']

except KeyboardInterrupt:
    print('Exiting from training early')

# Plot the training graph
f = open(model.hyper_params['log_file'])
lines = f.readlines()
lines.reverse()

train = []
test = []

for line in lines:
    if line[:10] == 'Simulation' and len(train) > 1:
        break
    elif line[:10] == 'Simulation' and len(train) <= 1:
        train, test = [], []

    if line[2:5] == 'end' and line[-5:-2] == 'VAL':
        test.append(line.strip().split("|"))
    elif line[2:5] == 'end' and line[-7:-2] == 'TRAIN':
        train.append(line.strip().split("|"))

train.reverse()
test.reverse()

train_ndcg = []
test_ndcg = []
test_loss, train_loss = [], []

for i in train:
    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            train_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            train_loss.append(float(metric.split("=")[1].split(' ')[1]))

total, avg_runtime = 0.0, 0.0

for i in test:
    avg_runtime += float(i[2].split(" ")[2][:-1])
    total += 1.0

    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            test_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            test_loss.append(float(metric.split("=")[1].split(' ')[1]))

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_title(hyper_params["project_name"], fontweight="bold", size=20)
ax1.plot(test_ndcg, 'b-')
ax1.set_xlabel('Epochs', fontsize=20.0)
ax1.set_ylabel('NDCG@100', color='b', fontsize=20.0)
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(test_loss, 'r--')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
if not os.path.isdir("saved_plots/"):
    os.mkdir("saved_plots/")
fig.savefig("saved_plots/learning_curve_" + hyper_params["project_name"] + ".png")
plt.show()

# Log figure
experiment.log_figure(figure=plt)

# Checking metrics for the test set on best saved model
with open(hyper_params['model_file_name'], 'rb') as f:
    model = torch.load(f)

# Calculate the metrics on the test set
metrics, len_to_ndcg_at_100_map = evaluate(model, criterion, test_reader, hyper_params, False, experiment)

# Plot sequence length vs NDCG@100 graph
plot_len_vs_ndcg(len_to_ndcg_at_100_map, experiment)

string = ""
for m in metrics:
    string += " | " + m + ' = ' + str(metrics[m])

ss = '=' * 89
ss += '\n| End of training'
ss += string + " (TEST)"
ss += '\n'
ss += '=' * 89
file_write(hyper_params['log_file'], ss)
print("average runtime per epoch =", round(avg_runtime / float(total), 4), "s")
