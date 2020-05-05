import os

# Dictionary to store model hyper-parameters
hyper_params = {
    'project_name': 'svae_ml1m',
    'model_file_name': '',
    'log_file': '',
    'history_split_test': [0.8, 0.2],  # Part of test history to train on : Part of test history to test
    'learning_rate': 0.01,  # learning rate is required only if optimizer is AdaGrad
    'optimizer': 'adam',
    'weight_decay': float(5e-3),
    'epochs': 50,
    'batch_size': 1,  # Needs to be 1, because we don't pack multiple sequences in the same batch
    'item_embed_size': 256,
    'rnn_size': 200,
    'hidden_size': 150,
    'latent_size': 64,
    'loss_type': 'next_k',  # [predict_next, same, prefix, postfix, exp_decay, next_k]
    'next_k': 4,
    'number_users_to_keep': 1000000000,
    'batch_log_interval': 1000,
    'train_cp_users': 200,
    'exploding_clip': 0.25,
}

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