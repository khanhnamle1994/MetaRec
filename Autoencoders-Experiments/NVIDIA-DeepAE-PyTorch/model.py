from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers


def Deep_AE_model(X, layers, activation, last_activation, dropout, regularizer_encode,
                  regularizer_decode, side_infor_size=0):
    """
    Function to build the deep autoencoders for collaborative filtering
    :param X: the given user-item interaction matrix
    :param layers: list of layers (each element is the number of neurons per layer)
    :param activation: choice of activation function for all dense layer except the last
    :param last_activation: choice of activation function for the last dense layer
    :param dropout: dropout rate
    :param regularizer_encode: regularizer for the encoder
    :param regularizer_decode: regularizer for the decoder
    :param side_infor_size: size of the one-hot encoding vector for side information
    :return: Keras model
    """

    # Input
    input_layer = x = Input(shape=(X.shape[1],), name='UserRating')

    # Encoder Phase
    k = int(len(layers) / 2)
    i = 0
    for l in layers[:k]:
        x = Dense(l, activation=activation,
                  name='EncLayer{}'.format(i),
                  kernel_regularizer=regularizers.l2(regularizer_encode))(x)
        i = i + 1

    # Latent Space
    x = Dense(layers[k], activation=activation,
              name='LatentSpace',
              kernel_regularizer=regularizers.l2(regularizer_encode))(x)

    # Dropout
    x = Dropout(rate=dropout)(x)

    # Decoder Phase
    for l in layers[k + 1:]:
        i = i - 1
        x = Dense(l, activation=activation,
                  name='DecLayer{}'.format(i),
                  kernel_regularizer=regularizers.l2(regularizer_decode))(x)

    # Output
    output_layer = Dense(X.shape[1] - side_infor_size, activation=last_activation, name='UserScorePred',
                         kernel_regularizer=regularizers.l2(regularizer_decode))(x)

    # This model maps an input to its reconstruction
    model = Model(input_layer, output_layer)

    return model
