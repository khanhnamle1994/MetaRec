import os
import numpy as np
import argparse
import torch

from Params import Params
from Dataset import Dataset
from Logger import Logger
from Evaluator import Evaluator
from Trainer import Trainer
from ModelBuilder import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CDAE')
parser.add_argument('--data_dir', type=str, default='../../')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--config_dir', type=str, default='./config')
parser.add_argument('--seed', type=int, default=1994)

config = parser.parse_args()
model_config = Params(os.path.join(config.config_dir, config.model.lower() + '.json'))
model_config.update_dict('exp_conf', config.__dict__)

np.random.seed(config.seed)
torch.random.manual_seed(config.seed)
device = torch.device('cpu')

dataset = Dataset(
    data_dir=config.data_dir,
    data_name=model_config.data_name,
    train_ratio=model_config.train_ratio,
    device=device
)

log_dir = os.path.join('saves', config.model)
logger = Logger(log_dir)
model_config.save(os.path.join(logger.log_dir, 'config.json'))

eval_pos, eval_target = dataset.eval_data()
item_popularity = dataset.item_popularity
evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_config.top_k)

model = build_model(config.model, model_config, dataset.num_users, dataset.num_items, device)

logger.info(model_config)
logger.info(dataset)

trainer = Trainer(
    dataset=dataset,
    model=model,
    evaluator=evaluator,
    logger=logger,
    conf=model_config
)

trainer.train()
