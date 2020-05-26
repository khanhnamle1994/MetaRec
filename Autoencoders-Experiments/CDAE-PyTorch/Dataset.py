import os
import numpy as np
import torch
from DataUtils import preprocess, load_data


class Dataset:
    def __init__(self, data_dir, data_name, train_ratio, device):
        self.train_ratio = train_ratio
        self.num_negatives = 3
        self.device = device

        if data_name == 'ml-100k':
            sep = '\t'
            filename = 'u.data'
            self.num_users, self.num_items = 943, 1682
        elif data_name == 'ml-1m':
            sep = '::'
            filename = 'ratings.dat'
            self.num_users, self.num_items = 6040, 3952
        else:
            raise NotImplementedError('Choose correct dataset: {ml-100k, ml-1m}')

        data_path = os.path.join(data_dir, data_name, data_name + '.data')
        stat_path = os.path.join(data_dir, data_name, data_name + '.stat')

        if os.path.exists(data_path) and os.path.exists(data_path):
            print('Already preprocessed. Load from file.')
        else:
            preprocess(os.path.join(data_dir, data_name, filename), data_path, stat_path, sep)

        print('Read movielens data from %s' % data_path)
        self.train_matrix, self.test_matrix, self.user_id_map, self.user_popularity, self.item_id_map, self.item_popularity, self.num_uesrs, self.num_items = load_data(
            data_path)

    def sparse_to_dict(self, sparse_matrix):
        ret_dict = {}
        num_users = sparse_matrix.shape[0]
        for u in range(num_users):
            items_u = sparse_matrix.indices[sparse_matrix.indptr[u]: sparse_matrix.indptr[u + 1]]
            ret_dict[u] = items_u.tolist()
        return ret_dict

    def eval_data(self):
        return self.train_matrix, self.sparse_to_dict(self.test_matrix)

    def generate_pairwise_data_from_matrix(self, rating_matrix, num_negatives=1, p=None):
        num_users, num_items = rating_matrix.shape

        users = []
        positives = []
        negatives = []
        for user in range(num_users):
            if p is None:
                start = rating_matrix.indptr[user]
                end = rating_matrix.indptr[user + 1]
                pos_index = rating_matrix.indices[start:end]
                num_positives = len(pos_index)
                if num_positives == 0:
                    print('[WARNING] user %d has 0 ratings. Not generating negative samples.' % user)
                    continue

                num_all_negatives = num_items - num_positives
                prob = np.full(num_items, 1 / num_all_negatives)
                prob[pos_index] = 0.0

            neg_items = np.random.choice(num_items, num_positives * num_negatives, replace=True, p=prob)
            for i, pos in enumerate(pos_index):
                users += [user] * num_negatives
                positives += [pos] * num_negatives
                negatives += neg_items[i * num_negatives: (i + 1) * num_negatives].tolist()

        return torch.LongTensor(users), torch.LongTensor(positives), torch.LongTensor(negatives)

    def __str__(self):
        # return string representation of 'Dataset' class
        # print(Dataset) or str(Dataset)
        ret = '======== [Dataset] ========\n'
        # ret += 'Train file: %s\n' % self.train_file
        # ret += 'Test file : %s\n' % self.test_file
        ret += 'Number of Users : %d\n' % self.num_users
        ret += 'Number of items : %d\n' % self.num_items
        ret += 'Split ratio: %s\n' % str(self.train_ratio)
        ret += '\n'
        return ret