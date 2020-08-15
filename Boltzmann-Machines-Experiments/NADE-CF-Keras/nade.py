# Import libraries
import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers


# NADE architecture
class NADE(Layer):
    def __init__(self, hidden_dim, activation, W_regularizer=None, V_regularizer=None,
                 b_regularizer=None, c_regularizer=None, bias=False, args=None, **kwargs):

        self.init = initializers.get('uniform')

        self.bias = bias
        self.activation = activation
        self.hidden_dim = hidden_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.c_regularizer = regularizers.get(c_regularizer)

        self.args = args

        super(NADE, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the NADE architecture
        :param input_shape: Shape of the input
        """
        self.input_dim1 = input_shape[1]
        self.input_dim2 = input_shape[2]

        self.W = self.add_weight(shape=(self.input_dim1, self.input_dim2, self.hidden_dim),
                                 initializer=self.init, name='{}_W'.format(self.name), regularizer=self.W_regularizer)

        if self.bias:
            self.c = self.add_weight(shape=(self.hidden_dim,), initializer=self.init,
                                     name='{}_c'.format(self.name), regularizer=self.c_regularizer)

        if self.bias:
            self.b = self.add_weight(shape=(self.input_dim1, self.input_dim2), initializer=self.init,
                                     name='{}_b'.format(self.name), regularizer=self.b_regularizer)

        self.V = self.add_weight(shape=(self.hidden_dim, self.input_dim1, self.input_dim2),
                                 initializer=self.init, name='{}_V'.format(self.name), regularizer=self.V_regularizer)

        super(NADE, self).build(input_shape)

    def call(self, original_x):
        args = self.args

        x = K.cumsum(original_x[:, :, ::-1], axis=2)[:, :, ::-1]
        # x.shape = (?,6040,5)
        # W.shape = (6040, 5, 500)
        # c.shape = (500,)
        output_ = tf.tensordot(x, self.W, axes=[[1, 2], [0, 1]])
        if args.normalize_1st_layer:
            output_ /= tf.matmul(
                tf.maximum(
                    tf.reshape(
                        tf.reduce_sum(
                            tf.reduce_sum(original_x, axis=2), axis=1),
                        [-1, 1]), 1), tf.ones([1, output_.shape[1]]))

        if self.bias:
            output_ = output_ + self.c

        h_out = tf.reshape(output_, [-1, self.hidden_dim])
        # tf.cast(indices, tf.float32)
        # output_.shape = (?,500)

        h_out_act = K.tanh(h_out)
        # h_out_act.shape = (?,500)
        # V.shape = (500, 6040, 5)
        # b.shape = (6040,5)

        if self.bias:
            output = tf.tensordot(h_out_act, self.V, axes=[[1], [0]]) + self.b
        else:
            output = tf.tensordot(h_out_act, self.V, axes=[[1], [0]])
        # output.shape = (?,6040,5)
        output = tf.reshape(output, [-1, self.input_dim1, self.input_dim2])
        return output

    def compute_output_shape(self, input_shape):
        """
        Compute the shape of the output
        :param input_shape: Shape of the input
        :return: Shape of the output
        """
        return input_shape[0], input_shape[1], input_shape[2]
