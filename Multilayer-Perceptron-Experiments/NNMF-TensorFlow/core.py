import tensorflow as tf

from model import NNMF
import dataset
from config import *


def _get_batch(train_data, batch_size):
    """
    Function to get current batch
    :param train_data: Training data
    :param batch_size: Batch size
    :return: training data sampled in batches
    """
    if batch_size:
        return train_data.sample(batch_size)
    return train_data


def _train(model, sess, saver, train_data, valid_data, batch_size):
    """
    Main training function
    :param model: Current model
    :param sess: Current TensorFlow session
    :param saver: Current TensorFlow saver
    :param train_data: Training data
    :param valid_data: Validation data
    :param batch_size: Batch size
    """
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0

    # Iteration loop
    for i in range(MAX_ITER):
        # Keep track of current epoch
        print("Epoch: {}".format(i))

        # Run Stochastic Gradient Descent for each batch
        batch = train_data.sample(batch_size) if batch_size else train_data
        model.train_iteration(batch)

        # Evaluate loss and RMSE for training and validation set
        train_loss = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_loss = model.eval_loss(valid_data)
        valid_rmse = model.eval_rmse(valid_data)
        print("Train Loss: {:3f}, Train RMSE: {:3f}, Valid Loss: {:3f}, Valid RMSE: {:3f}".format(
            train_loss, train_rmse, valid_loss, valid_rmse))

        # Special case with early stopping
        if EARLY_STOP:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(sess, model.model_file_path)
            elif early_stop_iters >= EARLY_STOP_MAX_ITER:
                print("Early stopping (Previous Valid RMSE: {} vs. Current Valid RMSE: {})...".format(
                    prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(sess, model.model_file_path)


def _test(model, valid_data, test_data):
    """
    Function to evaluate model on validation and test set
    :param model: Current model
    :param valid_data: Validation data
    :param test_data: Test data
    :return: RMSE for validation and test sets
    """
    valid_rmse = model.eval_rmse(valid_data)
    test_rmse = model.eval_rmse(test_data)
    print("Final Valid RMSE: {} and Final Test RMSE: {}".format(valid_rmse, test_rmse))
    return valid_rmse, test_rmse


def run(batch_size=None, **hyper_params):
    """Main execution loop"""
    kind = dataset.ML_1M

    with tf.compat.v1.Session() as sess:
        print("Reading in data")
        # Load data
        data = dataset.load_data(kind)

        print('Building network & initializing variables')
        # Initialize NNMF model
        model = NNMF(kind, **hyper_params)
        # Define computation graph
        model.init_sess(sess)
        # Define computation saver
        saver = tf.compat.v1.train.Saver()

        # Call training function
        _train(model, sess, saver, data['train'], data['valid'], batch_size=batch_size)

        print('Loading best checkpointed model')
        saver.restore(sess, model.model_file_path)
        # Return RMSE for validation and test sets
        valid_rmse, test_rmse = _test(model, data['valid'], data['test'])
        return valid_rmse, test_rmse