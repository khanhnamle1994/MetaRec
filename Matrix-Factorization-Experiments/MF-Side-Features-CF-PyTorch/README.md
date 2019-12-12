# Matrix Factorization for Collaborative Filtering with Side Features

This is my PyTorch implementation of a Matrix Factorization model for Collaborative Filtering. The code is inspired by [Chris Moody's tutorial on Deep Recommendations in PyTorch](https://docs.google.com/presentation/d/1gv7osHoSX8CHf0uzKSqOlxmmAvPPdmstL0nrZHWiHQM/edit#slide=id.p). This model adds in "side" features, especially useful in cold-start situations.

## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/data.py): This is the script that pre-processes the dataset.
* [loader.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/loader.py): This is the script that loads the dataset.
* [MFSideFeat.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/MFSideFeat.py): This is the model script that defines the Matrix Factorization model with side features.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/results). After training the model for 40 epochs, I got the training loss MSE = 0.619 and validation accuracy = 0.788.

## Run Tensorboard in the background.
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

<p float="left">
  <img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/loss_mse.png" width="500" />
  <img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Side-Features-CF-PyTorch/valid_accuracy.png" width="500" />
</p>
