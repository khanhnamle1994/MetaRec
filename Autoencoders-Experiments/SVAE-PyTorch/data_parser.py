import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from utils import file_write


def load_data(hyper_params):
    """Function to load the processed data"""
    file_write(hyper_params['log_file'], "Started reading data file")
    # Load train data
    f = open('processed_data/' + 'train.csv')
    lines_train = f.readlines()[1:]
    # Load validation train data
    f = open('processed_data/' + 'validation_tr.csv')
    lines_val_tr = f.readlines()[1:]
    # Load validation test data
    f = open('processed_data/' + 'validation_te.csv')
    lines_val_te = f.readlines()[1:]
    # Load test train data
    f = open('processed_data/' + 'test_tr.csv')
    lines_test_tr = f.readlines()[1:]
    # Load test test data
    f = open('processed_data/' + 'test_te.csv')
    lines_test_te = f.readlines()[1:]

    # Store the unique items into a text file
    unique_sid = list()
    with open('processed_data/' + 'unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    num_items = len(unique_sid)
    # Update log file
    file_write(hyper_params['log_file'], "Data Files loaded!")

    # Initialize Reader classes for train, validation, and test data
    train_reader = DataReader(hyper_params, lines_train, None, num_items, True)
    val_reader = DataReader(hyper_params, lines_val_tr, lines_val_te, num_items, False)
    test_reader = DataReader(hyper_params, lines_test_tr, lines_test_te, num_items, False)

    return train_reader, val_reader, test_reader, num_items


LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


class DataReader:
    """
    Class to read the data
    """
    def __init__(self, hyper_params, a, b, num_items, is_training):
        """
        Function to initialize the class
        :param hyper_params: Hyper-parameter dictionary
        :param a: Variable to store the train set
        :param b: Variable to store the test set
        :param num_items: Number of items
        :param is_training: Boolean value to check training mode
        """
        self.hyper_params = hyper_params
        self.batch_size = hyper_params['batch_size']

        num_users = 0
        min_user = 1000000000000000000000000  # Infinity
        for line in a:
            line = line.strip().split(",")
            num_users = max(num_users, int(line[0]))
            min_user = min(min_user, int(line[0]))
        num_users = num_users - min_user + 1

        self.num_users = num_users
        self.min_user = min_user
        self.num_items = num_items

        self.data_train = a
        self.data_test = b
        self.is_training = is_training
        self.all_users = []

        # Prepare the data
        self.prep()
        self.number()

    def prep(self):
        """
        Function to prepare the data
        """
        self.data = []
        for i in range(self.num_users):
            self.data.append([])

        for i in tqdm(range(len(self.data_train))):
            line = self.data_train[i]
            line = line.strip().split(",")
            self.data[int(line[0]) - self.min_user].append([int(line[1]), 1])

        if not self.is_training:
            self.data_te = []
            for i in range(self.num_users):
                self.data_te.append([])

            for i in tqdm(range(len(self.data_test))):
                line = self.data_test[i]
                line = line.strip().split(",")
                self.data_te[int(line[0]) - self.min_user].append([int(line[1]), 1])

    def number(self):
        self.num_b = int(min(len(self.data), self.hyper_params['number_users_to_keep']) / self.batch_size)

    def iter(self):
        users_done = 0

        x_batch = []

        user_iterate_order = list(range(len(self.data)))

        # Randomly shuffle the training order
        np.random.shuffle(user_iterate_order)

        for user in user_iterate_order:

            if users_done > self.hyper_params['number_users_to_keep']:
                break
            users_done += 1

            y_batch_s = torch.zeros(self.batch_size, len(self.data[user]) - 1, self.num_items)

            if self.hyper_params['loss_type'] == 'predict_next':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in [self.data[user][timestep + 1]]]), 1.0
                    )

            elif self.hyper_params['loss_type'] == 'next_k':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in self.data[user][timestep + 1:][:self.hyper_params['next_k']]]), 1.0
                    )

            elif self.hyper_params['loss_type'] == 'postfix':
                for timestep in range(len(self.data[user]) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in self.data[user][timestep + 1:]]), 1.0
                    )

            x_batch.append([i[0] for i in self.data[user][:-1]])

            if len(x_batch) == self.batch_size:  # batch_size always = 1

                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False)
                x_batch = []

    def iter_eval(self):

        x_batch = []
        test_movies, test_movies_r = [], []

        users_done = 0

        for user in range(len(self.data)):

            users_done += 1
            if users_done > self.hyper_params['number_users_to_keep']:
                break

            if self.is_training:
                split = float(self.hyper_params['history_split_test'][0])
                base_predictions_on = self.data[user][:int(split * len(self.data[user]))]
                heldout_movies = self.data[user][int(split * len(self.data[user])):]
            else:
                base_predictions_on = self.data[user]
                heldout_movies = self.data_te[user]

            y_batch_s = torch.zeros(self.batch_size, len(base_predictions_on) - 1, self.num_items)

            if self.hyper_params['loss_type'] == 'predict_next':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in [base_predictions_on[timestep + 1]]]), 1.0
                    )

            elif self.hyper_params['loss_type'] == 'next_k':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in base_predictions_on[timestep + 1:][:self.hyper_params['next_k']]]),
                        1.0
                    )

            elif self.hyper_params['loss_type'] == 'postfix':
                for timestep in range(len(base_predictions_on) - 1):
                    y_batch_s[len(x_batch), timestep, :].scatter_(
                        0, LongTensor([i[0] for i in base_predictions_on[timestep + 1:]]), 1.0
                    )

            test_movies.append([i[0] for i in heldout_movies])
            test_movies_r.append([i[1] for i in heldout_movies])
            x_batch.append([i[0] for i in base_predictions_on[:-1]])

            if len(x_batch) == self.batch_size:  # batch_size always = 1

                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False), \
                      test_movies, test_movies_r

                x_batch = []
                test_movies, test_movies_r = [], []
