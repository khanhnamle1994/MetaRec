An Implementation of [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) (Liang et al. 2018) in PyTorch.

<img src="https://raw.githubusercontent.com//khanhnamle1994/transfer-rec/tree/master/VAE-CF-PyTorch/pics/vae.png" width="500">

<img src="https://raw.githubusercontent.com//khanhnamle1994/transfer-rec/tree/master/VAE-CF-PyTorch/pics/result.png" width="500">

This repository gives you an implementation of the Vartiaonal Autoencoders for Collaborative Filtering model in PyTorch. Every data preprocessing step and code follows exactly from the [authors' Repo](https://github.com/dawenl/vae_cf).

# Requirements

```
PyTorch 0.4 & Python 3.6
Numpy
TensorboardX
```

# Examples

`python main.py --cuda` for full training.

# Dataset

You should execute `python data.py` first to download necessary data and preprocess MovieLens-20M dataset.

<img src="https://raw.githubusercontent.com//khanhnamle1994/transfer-rec/tree/master/VAE-CF-PyTorch/pics/data.png" width="500">

# Results

<img src="https://raw.githubusercontent.com//khanhnamle1994/transfer-rec/tree/master/VAE-CF-PyTorch/pics/result-experiment.png" width="500">
