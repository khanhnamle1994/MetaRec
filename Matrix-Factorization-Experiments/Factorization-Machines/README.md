# Factorization Machine for Collaborative Filtering

This is my PyTorch implementation of a Factorization Machine model for Collaborative Filtering. This model upgrades the core of Matrix Factorization to Factorization Machines, which enables a huge number of interactions while keeping computation under control.

## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Scripts
* [FM.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/FM-CF-PyTorch/FM.py): This is the model script that defines the Factorization Machine model.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/FM-CF-PyTorch/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/FM-CF-PyTorch/results). After training the model for 50 epochs, I got the training loss MSE = 0.658 and validation accuracy = 0.821.

## Run Tensorboard in the background.
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/FM-CF-PyTorch/loss_mse.png" width="500" /><img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/FM-CF-PyTorch/valid_accuracy.png" width="500" />
