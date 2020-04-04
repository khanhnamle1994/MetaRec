import tensorflow as tf

from model import NNMF
import dataset
from config import *


def _get_batch(train_data, batch_size):
    if batch_size:
        return train_data.sample(batch_size)
    return train_data


def _train(model, sess, saver, train_data, valid_data, batch_size):
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0
    for i in range(MAX_ITER):
        print("Epoch: {}".format(i))

        # Run SGD
        batch = train_data.sample(batch_size) if batch_size else train_data
        model.train_iteration(batch)

        # Evaluate
        train_loss = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_loss = model.eval_loss(valid_data)
        valid_rmse = model.eval_rmse(valid_data)
        print("Train Loss: {:3f}, Train RMSE: {:3f}, Valid Loss: {:3f}, Valid RMSE: {:3f}".format(
            train_loss, train_rmse, valid_loss, valid_rmse))

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
    valid_rmse = model.eval_rmse(valid_data)
    test_rmse = model.eval_rmse(test_data)
    print("Final Valid RMSE: {} and Final Test RMSE: {}".format(valid_rmse, test_rmse))
    return valid_rmse, test_rmse


def run(batch_size=None, **hyper_params):
    kind = dataset.ML_1M

    with tf.compat.v1.Session() as sess:
        # Process data
        print("Reading in data")
        data = dataset.load_data(kind)

        # Define computation graph & Initialize
        print('Building network & initializing variables')
        model = NNMF(kind, **hyper_params)
        model.init_sess(sess)
        saver = tf.compat.v1.train.Saver()

        _train(model, sess, saver, data['train'], data['valid'], batch_size=batch_size)

        print('Loading best checkpointed model')
        saver.restore(sess, model.model_file_path)
        valid_rmse, test_rmse = _test(model, data['valid'], data['test'])
        return valid_rmse, test_rmse