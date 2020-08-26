# Import libraries
import torch
import random

# Import utility script
from options import config, states


def training(melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    """
    Train MeLU
    :param melu: MeLU mode
    :param total_dataset: given dataset
    :param batch_size: batch size
    :param num_epoch: number of epochs
    :param model_save: boolean whether to save model
    :param model_filename: name of model
    :return: saved MeLU model
    """
    training_set_size = len(total_dataset)
    melu.train()
    for _ in range(num_epoch):
        # Shuffle the data
        random.shuffle(total_dataset)
        # Number of batches
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                # Generate support and query sets
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
            except IndexError:
                continue
            # Perform global update
            melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

    # Save the model
    if model_save:
        torch.save(melu.state_dict(), model_filename)
