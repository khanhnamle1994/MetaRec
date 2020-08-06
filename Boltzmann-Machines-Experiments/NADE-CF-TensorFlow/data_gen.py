# Import packages
import numpy as np
import json
import random
from itertools import islice
import keras.callbacks


class DataSet(keras.callbacks.Callback):
    """
    A data generator class that feeds data from given files.
    """
    def __init__(self, file_list, num_users, num_items, batch_size, mode, shuffle=True):
        self.flist = file_list
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle

    def get_corpus_size(self):
        """
        Computes and returns the number of samples in the corpus.
        """
        line_count = 0
        for dfile in self.flist:
            with open(dfile) as f:
                for i, l in enumerate(f):
                    pass
            line_count += (i + 1)

        self.size = line_count

        return self.size

    def generate(self, max_iters=-1):
        iter_cnt = 0
        while True:
            for dfile in self.flist:
                with open(dfile) as df:
                    while True:
                        next_n_data_lines = list(islice(df, self.batch_size))

                        if not next_n_data_lines:
                            break

                        self.input_ranking_vectors = np.zeros((self.batch_size, self.num_users, 5), dtype='int8')
                        self.output_ranking_vectors = np.zeros((self.batch_size, self.num_users, 5), dtype='int8')
                        self.input_mask_vectors = np.zeros((self.batch_size, self.num_users), dtype='int8')
                        self.output_mask_vectors = np.zeros((self.batch_size, self.num_users), dtype='int8')

                        for i, line in enumerate(next_n_data_lines):
                            line = json.loads(line)
                            movie_id = line['movieId']
                            rankings = line['rankings']

                            user_ids = []
                            values = []
                            flags = []
                            for ranking in rankings:
                                user_ids.append(int(ranking['userId']))
                                values.append(int(ranking['value']))
                                flags.append(int(ranking['flag']))

                            if self.mode == 0:
                                ordering = np.random.permutation(np.arange(len(user_ids)))
                                d = np.random.randint(0, len(ordering))
                                flag_in = (ordering < d)
                                flag_out = (ordering >= d)

                                self.input_mask_vectors[i][user_ids] = flag_in
                                self.output_mask_vectors[i][user_ids] = flag_out

                                if self.shuffle:
                                    shuffle_list = list(zip(user_ids, values))
                                    random.shuffle(shuffle_list)
                                    user_ids, values = zip(*shuffle_list)
                                for j, (user_id, value) in enumerate(zip(user_ids, values)):
                                    if flag_in[j]:
                                        self.input_ranking_vectors[i, user_id, (value - 1)] = 1
                                    else:
                                        self.output_ranking_vectors[i, user_id, (value - 1)] = 1

                            elif self.mode == 1:
                                for j, (user_id, value, flag) in enumerate(zip(user_ids, values, flags)):
                                    if flag == 0:
                                        self.input_ranking_vectors[i, user_id, (value - 1)] = 1
                                    else:
                                        self.output_ranking_vectors[i, user_id, (value - 1)] = 1

                            elif self.mode == 2:
                                for j, (user_id, value, flag) in enumerate(zip(user_ids, values, flags)):
                                    if flag == 0:
                                        self.input_ranking_vectors[i, user_id, (value - 1)] = 1
                                    if flag == 1:
                                        self.input_ranking_vectors[i, user_id, (value - 1)] = 1
                                    else:
                                        self.output_ranking_vectors[i, user_id, (value - 1)] = 1

                        inputs = {
                            'input_ratings': self.input_ranking_vectors,
                            'output_ratings': self.output_ranking_vectors,
                            'input_masks': self.input_mask_vectors,
                            'output_masks': self.output_mask_vectors,
                        }

                        outputs = {'nade_loss': np.zeros([self.batch_size])}
                        yield (inputs, outputs)

            if self.shuffle:
                iter_cnt += 1
                if max_iters != -1:
                    if iter_cnt == max_iters:
                        break

                print('shuffling data...')
                random.shuffle(self.flist)
