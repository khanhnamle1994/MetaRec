# Matrix Factorization for Collaborative Filtering

This is a series of Matrix Factorization models for Collaborative Filtering implemented in PyTorch. The code is heavily inspired by [Chris Moody's tutorial on Deep Recommendations in PyTorch](https://docs.google.com/presentation/d/1gv7osHoSX8CHf0uzKSqOlxmmAvPPdmstL0nrZHWiHQM/edit#slide=id.p).

Here are the 7 different models:

* [Vanilla Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Vanilla-MF)
* [Matrix Factorization with Biases](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Biases)
* [Matrix Factorization with Side Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Side-Features)
* [Matrix Factorization with Temporal Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Temporal-Features)
* [Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Factorization-Machines)
* [Matrix Factorization with Mixture of Tastes](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Mixture-Tastes)
* [Vanilla Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Variational-MF)

## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Processing and Loading Data
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/data.py): This is the script that pre-processes the data.
* [loader.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/loader.py): This is the script that loads the data.

## Training Models

## Evaluating Results