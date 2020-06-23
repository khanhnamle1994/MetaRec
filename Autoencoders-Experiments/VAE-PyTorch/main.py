# CometML for experiment logging
from comet_ml import Experiment

# Import packages
import os
import numpy as np
import argparse
import torch

# Import utility scripts
from Params import Params
from Dataset import Dataset
from Logger import Logger
from Evaluator import Evaluator
from Trainer import Trainer
from ModelBuilder import build_model

# Create an experiment with API key
experiment = Experiment(api_key="GaicgDHvizDRCbpq2wVV8NHnX", project_name="autoencoders-movielens1M")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MultVAE')
parser.add_argument('--data_dir', type=str, default='../../')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--config_dir', type=str, default='./config')
parser.add_argument('--seed', type=int, default=1994)

config = parser.parse_args()
# Get model configuration
model_config = Params(os.path.join(config.config_dir, config.model.lower() + '.json'))
model_config.update_dict('exp_conf', config.__dict__)
# Set random seeds
np.random.seed(config.seed)
torch.random.manual_seed(config.seed)
# Device defaulted to CPU
device = torch.device('cpu')

# Initialize Dataset class
dataset = Dataset(
    data_dir=config.data_dir,
    data_name=model_config.data_name,
    train_ratio=model_config.train_ratio,
    device=device
)

log_dir = os.path.join('saves', config.model)
# Initialize Logger class
logger = Logger(log_dir)
model_config.save(os.path.join(logger.log_dir, 'config.json'))
# Get the position and target of the evaluated item
eval_pos, eval_target = dataset.eval_data()
# Get the popularity item
item_popularity = dataset.item_popularity
# Initialize Evaluator class
evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_config.top_k)
# Build the model
model = build_model(config.model, model_config, dataset.num_users, dataset.num_items, device)
# Get the model info and data info
logger.info(model_config)
logger.info(dataset)

# Initialize Trainer class
trainer = Trainer(
    dataset=dataset,
    model=model,
    evaluator=evaluator,
    logger=logger,
    conf=model_config
)
# Train the model and get results
trainer.train(experiment)

# Dictionary to store model hyper-parameters
hyper_params = {
    "encoding_dims": 200,
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000,
    "epochs": 500,
    "batch_size": 512,
    "optimizer": "Adam",
    "learning_rate": 0.01
}

# Log model hyper-parameters
experiment.log_parameters(hyper_params)
