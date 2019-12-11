# Neural Collaborative Filtering

This is my PyTorch implementation of the paper [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). There are 3 collaborative filtering models implemented: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, these models were optimized using log loss with negative sampling. Every data preprocessing step and code follows exactly from [the authors' repo](https://github.com/hexiangnan/neural_collaborative_filtering).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/pics/Fig2.png" width="500">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/data.py): This is the data processing script.
* [engine.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/engine.py): This is the meta engine script for training and evaluating the models.
* [metrics.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/metrics.py): This is the metric script that defines the Hit Ratio metric and Normalized Discounted Cumulative Gain metric.
* [utils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/utils.py): This is the utility script that define some handy functions for PyTorch model training.
* [gmf.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/gmf.py): This is the model script that defines the General Matrix Factorization model.
* [mlp.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/mlp.py): This is the model script that defines the Multi-Layer Perceptron model.
* [neumf.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/neumf.py): This is the model script that defines the Neural Matrix Factorization model.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/train.py): This is the main training script. You can simply run `python train.py` to execute it.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/pics/Fig3.png" width="500">

## Requirements

```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
TensorboardX
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/pics/Table1.png" width="500">

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Neural-CF-PyTorch/results). All models are trained for 199 steps. Here are the results after the last iteration:
* General Matrix Factorization: Loss = 5186, Hit Ratio = 0.631, NDCG = 0.366
* Multi-Layer Perceptron: Loss = 3171, Hit Ratio = 0.500, NDCG = 0.275
* Neural Matrix Factorization: Loss = 2431, Hit Ratio = 0.730, NDCG = 0.447

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Neural-CF-PyTorch/pics/result-experiment.png" width="700">
