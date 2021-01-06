# Matrix Factorization for Collaborative Filtering with Biases

This is my PyTorch implementation of a Matrix Factorization model for Collaborative Filtering.
This model includes biases for extra predictive power.

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
* [loader.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Biases/loader.py): This is the script that loads the data.
* [MFBiases.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Biases/MFBiases.py): This is the model script that defines the Matrix Factorization model with biases.
* [train.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Biases/train.py): This is the main training script. You can simply run `python train.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/MetaRec/tree/master/Matrix-Factorization-Experiments/MF-Biases/results).
After training the model for 50 epochs with 75/25 train-test split, I got the training loss MSE = 0.6789 and test loss MSE = 0.7895 with training time = 11m38s.

## Run Tensorboard In The Background
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models.
It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

Here is the Mean Squared Error Loss on the training set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Biases/loss_mse.svg" width="1000" />

Here is the Mean Squared Error Loss on the test set:

<img src="https://github.com/khanhnamle1994/MetaRec/blob/master/Matrix-Factorization-Experiments/MF-Biases/validation_avg_loss.svg" width="1000" />
