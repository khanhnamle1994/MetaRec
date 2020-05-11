import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K


def show_error(history, skip):
    """
    Function to show model loss
    :param history: model log history
    :param skip: interval to be skipped
    :return: visualization of model loss
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Matplotlib settings
    plt.plot(np.arange(skip, len(loss), 1), loss[skip:])
    plt.plot(np.arange(skip, len(loss), 1), val_loss[skip:])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def show_rmse(history, skip):
    """
    Function to show the RMSE evaluation metric
    :param history: model log history
    :param skip: interval to be skipped
    :return: visualization of model RMSE
    """
    rmse = history.history['masked_rmse_clip']
    val_rmse = history.history['val_masked_rmse_clip']
    # Matplotlib settings
    plt.plot(np.arange(skip, len(rmse), 1), rmse[skip:])
    plt.plot(np.arange(skip, len(val_rmse), 1), val_rmse[skip:])
    plt.title('model train vs validation masked_rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


def load_model(name):
    """
    Function to load the model
    :param name: choice of model
    :return: loaded model
    """
    # Load JSON file and create model
    model_file = open('{}.json'.format(name), 'r')
    loaded_model_json = model_file.read()
    model_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights("{}.h5".format(name))
    print("Loaded model from disk")
    return loaded_model


def save_model(name, model):
    """
    Function to save the model
    :param name: choice of model
    :param model: given model
    """
    # Serialize model to JSON
    model_json = model.to_json()
    with open("{}.json".format(name), "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights("{}.h5".format(name))
    print("Saved model to disk")


def masked_se(y_true, y_pred):
    """
    Function to define the masked squared error
    :param y_true: true label
    :param y_pred: predicted label
    :return: masked squared error
    """
    # Masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # Masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1)
    return masked_mse


def masked_rmse(y_true, y_pred):
    """
    Function to define the masked root mean squared error
    :param y_true: true label
    :param y_pred: predicted label
    :return: masked root mean squared error
    """
    # Masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # Masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_rmse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_rmse


def masked_rmse_clip(y_true, y_pred):
    """
    Function to define the masked root mean squared error with clipping
    :param y_true: true label
    :param y_pred: predicted label
    :return: masked root mean squared error with clipping
    """
    # Masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 1, 5)
    # Masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_rmse_clip = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_rmse_clip
