# Matrix Factorization for Collaborative Filtering with Biases

This is my PyTorch implementation of a Matrix Factorization model for Collaborative Filtering. This model includes biases for extra predictive power.

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
* [MFBiases.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Biases-CF-PyTorch/MFBiases.py): This is the model script that defines the Matrix Factorization model with biases.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Biases-CF-PyTorch/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Biases-CF-PyTorch/results). After training the model for 50 epochs, I got the training loss MSE = 0.6623 and validation accuracy = 0.788 with training time = 12m 32s.

## Run Tensorboard in the background.
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

Here is the Mean Squared Error Loss for this model:

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Biases/loss_mse.svg" width="750" />

Here is the Accuracy on the Test set for this model:

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/MF-Biases/valid_avg_accuracy.svg" width="750" />
