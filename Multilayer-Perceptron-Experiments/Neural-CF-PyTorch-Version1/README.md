# Neural Collaborative Filtering (Version 1)

This is my PyTorch implementation of the paper [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017).
There are 3 collaborative filtering models implemented: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF).
To target the models for implicit feedback and ranking task, these models were optimized using log loss with negative sampling. Every data preprocessing step and code follows exactly from [the authors' repo](https://github.com/hexiangnan/neural_collaborative_filtering).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1/pics/Fig2.png" width="800">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/data.py): This is the data processing script.
* [engine.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/engine.py): This is the meta engine script for training and evaluating the models.
* [metrics.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/metrics.py): This is the metric script that defines the Hit Ratio metric and Normalized Discounted Cumulative Gain metric.
* [utils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/utils.py): This is the utility script that define some handy functions for PyTorch model training.
* [gmf.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/gmf.py): This is the model script that defines the General Matrix Factorization model.
* [mlp.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/mlp.py): This is the model script that defines the Multi-Layer Perceptron model.
* [neumf.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/neumf.py): This is the model script that defines the Neural Matrix Factorization model.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch-Version1/train.py): This is the main training script. You can simply run `python train.py` to execute it.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1/pics/Fig3.png" width="700">

## Requirements

```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
TensorboardX
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1/pics/Table1.png" width="600">

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Neural-CF-PyTorch-Version1/results). All models are trained for 50 epochs. Here are the results after the last iteration:
* General Matrix Factorization: Hit Ratio = 0.6397, NDCG = 0.3669
* Multi-Layer Perceptron: Hit Ratio = 0.6550, NDCG = 0.3796
* Neural Matrix Factorization: Hit Ratio = 0.6594, NDCG = 0.3919

## Run Tensorboard in the background.
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

Here are the Hit Ratio @ 10 performances:

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1/pics/performance_HR.svg" width="500">

Here are the Normalized Discounted Cumulative Gain @ 10 performances:

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1/pics/performance_NDCG.svg" width="500">