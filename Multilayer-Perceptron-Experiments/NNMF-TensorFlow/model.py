import os
import time
import math

import tensorflow as tf

from .dataset import get_N_and_M


def _init_model_file_path(kind):
    folder_path = 'logs/{}'.format(int(time.time() * 1000))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return os.path.join(folder_path, 'model.ckpt')


def _get_weight_init_range(n_in, n_out):
    """
        Calculates range for picking initial weight values from a uniform distribution.
    """
    return 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)


def _build_mlp(layer,
               training=False,
               hidden_unit_number=50,
               hidden_layer_number=3,
               output_unit_number=1,
               activation=tf.nn.sigmoid,
               final_activation=None):
    """
        Builds a feed-forward NN (MLP)
    """
    prev_layer_unit_number = layer.get_shape().as_list()[1]
    Ws, bs = [], []

    unit_numbers = [hidden_unit_number] * (hidden_layer_number - 1) + [output_unit_number]

    for i, unit_number in enumerate(unit_numbers):
        # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
        range = _get_weight_init_range(prev_layer_unit_number, unit_number)
        W = tf.Variable(
            tf.compat.v1.random_uniform([prev_layer_unit_number, unit_number], minval=-range, maxval=range))
        b = tf.Variable(tf.zeros([unit_number]))
        Ws.append(W)
        bs.append(b)

        layer = tf.matmul(layer, W) + b
        if i < len(unit_numbers) - 1:
            # layer = tf.layers.batch_normalization(layer, training=training)
            layer = activation(layer)
            # if dropout_rate > 0:
            #     layer = tf.layers.dropout(layer, rate=dropout_rate, training=training)
        else:
            if final_activation:
                layer = final_activation(layer)
        prev_layer_unit_number = unit_number

    return layer, Ws + bs


class NNMF(object):
    def __init__(self, kind, D=10, D_prime=60, K=1,
                 hidden_unit_number=50,
                 hidden_layer_number=3,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1},
                 lambda_value=50,
                 learning_rate=1e-3):
        self.lambda_value = lambda_value
        self.N, self.M = get_N_and_M(kind)
        self.D = D
        self.D_prime = D_prime
        self.K = K
        self.hidden_unit_number = hidden_unit_number
        self.latent_normal_init_params = latent_normal_init_params
        self.hidden_layer_number = hidden_layer_number
        self.model_file_path = _init_model_file_path(kind)
        # self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.user_index = tf.compat.v1.placeholder(tf.int32, [None])
        self.item_index = tf.compat.v1.placeholder(tf.int32, [None])
        self.r_target = tf.compat.v1.placeholder(tf.float32, [None])
        # self.timestamp = tf.compat.v1.placeholder(tf.float32, [None])

        # Call methods to initialize variables and operations (to be implemented by children)
        self._init_vars()
        self._init_ops()

        self.rmse = tf.sqrt(
            tf.reduce_mean(tf.square(tf.subtract(self.r, self.r_target))))

    def init_sess(self, sess):
        self.sess = sess
        init = tf.compat.v1.initialize_all_variables()
        self.sess.run(init)

    def _init_vars(self):
        self.training = tf.compat.v1.placeholder(tf.bool)

        # Latents
        self.U = tf.Variable(
            tf.compat.v1.truncated_normal([self.N, self.D], **
                                self.latent_normal_init_params))
        self.U_prime = tf.Variable(
            tf.compat.v1.truncated_normal([self.N, self.D_prime, self.K], **
                                self.latent_normal_init_params))
        self.V = tf.Variable(
            tf.compat.v1.truncated_normal([self.M, self.D], **
                                self.latent_normal_init_params))
        self.V_prime = tf.Variable(
            tf.compat.v1.truncated_normal([self.M, self.D_prime, self.K], **
                                self.latent_normal_init_params))

        # Lookups
        self.U_lookup = tf.nn.embedding_lookup(self.U, self.user_index)
        self.U_prime_lookup = tf.nn.embedding_lookup(self.U_prime,
                                                     self.user_index)
        self.V_lookup = tf.nn.embedding_lookup(self.V, self.item_index)
        self.V_prime_lookup = tf.nn.embedding_lookup(self.V_prime,
                                                     self.item_index)

        # MLP ("f")
        prime = tf.reduce_sum(
            tf.multiply(self.U_prime_lookup, self.V_prime_lookup), axis=2)
        f_input_layer = tf.concat(
            values=[self.U_lookup, self.V_lookup, prime], axis=1)

        # activation = tf.nn.sigmoid
        activation = tf.nn.relu

        final_activation = None
        # final_activation = lambda x: (tf.nn.sigmoid(x) * 4 + 1)
        # final_activation = lambda x: (tf.nn.tanh(x) * 2 + 3)

        _r, self.mlp_weights = _build_mlp(
            f_input_layer,
            self.training,
            hidden_unit_number=self.hidden_unit_number,
            hidden_layer_number=self.hidden_layer_number,
            output_unit_number=1,
            activation=activation,
            final_activation=final_activation)

        # self.r = _r
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        self.reconstruction_loss = tf.reduce_sum(
            tf.square(tf.subtract(self.r_target, self.r)),
            reduction_indices=[0])
        self.regularizer_loss = tf.add_n([
            tf.reduce_sum(tf.square(self.U_prime)),
            tf.reduce_sum(tf.square(self.U)),
            tf.reduce_sum(tf.square(self.V)),
            tf.reduce_sum(tf.square(self.V_prime)),
        ])
        self.loss = self.reconstruction_loss + (
            self.lambda_value * self.regularizer_loss)

        # Optimizer
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)

        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(
            self.loss, var_list=self.mlp_weights)
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(
            self.loss, var_list=[self.U, self.U_prime, self.V, self.V_prime])

        self.optimize_steps = [f_train_step, latent_train_step]

    def train_iteration(self, data, additional_feed=None):
        # import pandas as pd
        # print(pd.Series.min(data['timestamp']))
        # print(pd.Series.max(data['timestamp']))
        feed_dict = {
            self.user_index: data['user_id'],
            self.item_index: data['item_id'],
            self.r_target: data['rating'],
            # self.timestamp: data['timestamp'],
            self.training: True
        }

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._iters += 1

    def eval_loss(self, data):
        feed_dict = {
            self.user_index: data['user_id'],
            self.item_index: data['item_id'],
            self.r_target: data['rating'],
            # self.timestamp: data['timestamp'],
            self.training: False
        }
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(
            self.r,
            feed_dict={
                self.user_index: [user_id],
                self.item_index: [item_id],
                # self.timestamp: data['timestamp'],
                self.training: False
            })
        return rating[0]

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {
            self.user_index: user_ids,
            self.item_index: item_ids,
            self.r_target: ratings,
            # self.timestamp: data['timestamp'],
            self.training: False
        }
        return self.sess.run(self.rmse, feed_dict=feed_dict)