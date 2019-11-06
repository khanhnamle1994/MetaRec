import pandas as pd
import numpy as np
from rbm import RBMEngine
from data import SampleGenerator

rbm_config = {'alias': 'rbm_model',
            'num_epoch': 200,
            'batch_size': 512,
            'optimizer': 'adam',
            'adam_lr': 1e-3,
            'num_users': 6040,
            'num_items': 3706,
            'l2_regularization': 1e-4,
            'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

# Load Data
ml1m_dir = '/Users/khanhnamle/Desktop/CSCI799-Graduate-Independent-Study/Codebase/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep = '::', header = None, names = ['uid', 'mid', 'rating', 'timestamp'],  engine = 'python')

# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))

ml1m_rating = pd.merge(ml1m_rating, user_id, on = ['uid'], how = 'left')

item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))

ml1m_rating = pd.merge(ml1m_rating, item_id, on = ['mid'], how = 'left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

# DataLoader for training
sample_generator = SampleGenerator(ratings = ml1m_rating)
evaluate_data = sample_generator.evaluate_data

config = rbm_config
engine = RBMEngine(config)

for epoch in range(config['num_epoch']):

    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)

    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id = epoch)

    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id = epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)
