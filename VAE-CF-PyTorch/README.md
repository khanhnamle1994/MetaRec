# Variational Autoencoders for Collaborative Filtering

This is my PyTorch implementation of the paper [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara (2018). In particular, I trained a Variational Autoencoder with multinomial likelihood (described in the paper) on the public Movielens-20M dataset. Every data preprocessing step and code follows exactly from the [authors' Repo](https://github.com/dawenl/vae_cf).

## Requirements

```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
TensorboardX
```

## Dataset
You should execute `python data.py` first to download necessary data and preprocess MovieLens-20M dataset.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/VAE-CF-PyTorch/pics/data.png" width="500">

## VAE model
The file `models.py` contains an implementation of the Variational Autoencoder with Multinomial Likelihood.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/VAE-CF-PyTorch/pics/vae.png" width="700">

## Metrics
The file `metric.py` defines the metrics: recall@k and ndcg@k

## Training
You should execute `python main.py` for full training. Here are the hyperparameters that I use during training: Learning Rate = 1e-4, Batch Size = 500, Epoch = 200, Annealing Steps = 200000, Annealing Parameter = 0.2.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/VAE-CF-PyTorch/Results). I was able to reproduce the results from the original paper with training loss = 474, validation loss = 377, recall@20 = 0.54, and ndcg@100 = 0.43.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/VAE-CF-PyTorch/pics/result.png" width="500">

## Run Tensorboard in the background.
While I am using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
tensorboard --logdir runs
```

Visit the Tensorboard dashboard by going to [http://localhost:6006](http://localhost:6006)

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/VAE-CF-PyTorch/pics/result-experiment.png" width="700">
