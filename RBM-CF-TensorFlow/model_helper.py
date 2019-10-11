import tensorflow as tf

def _get_bias_init():
    return tf.zeros_initializer()

def _get_weight_init():
    return tf.random_normal_initializer(mean=0.0, stddev=0.5)
