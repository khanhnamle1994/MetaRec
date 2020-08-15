# A Neural Autoregressive Approach to Collaborative Filtering

This is my TensorFlow implementation of the paper [A Neural Autoregressive Approach to Collaborative Filtering](https://arxiv.org/abs/1605.09477) by Yin Zheng, Bangsheng Tang, Wenkui Ding, and Hanning Zhou (2016).
This approach is inspired by the [Restricted Boltzmann Machine based CF model](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/RBM-CF-PyTorch) and the [Neural Autoregressive Distribution Estimator](https://arxiv.org/abs/1605.02226).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/pics/NADE.png" width="700">

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

## Scripts
* [configs.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/configs.py): This is the configuration script that sets the data directory.
* [indexes.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/indexes.py): This the indexing script that collect user/item indicies and map them to recommendations.
* [data_prep.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/data_prep.py): This is the data processing script that uses PySpark to split the data into train (85.5%), validation (4.5%), and test (10%) sets.
* [data_gen.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/data_gen.py): This is the data generating script that uses a DataGenerator class to instantiate datasets.
* [nade.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/nade.py): This is the model script that defines the Neural Autoregressive Distribution Estimator model.
* [run.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/NADE-CF-Keras/run.py): This is the main running script that trains and evaluates the model on MovieLens 1M.

## Requirements
You need to activate a virtual environment to execute the code:

```
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
python data_prep.py
python run.py
```

## Results
The NADE architecture has 100 hidden units. Adam optimizer with a learning rate of 0.001 was used during training.
I obtained a Root Mean Squared Error (RMSE) value of 0.920 on the training set after training the model for 50 epochs.
The RMSE on the test set is 0.902. Training time takes 90m45s. 