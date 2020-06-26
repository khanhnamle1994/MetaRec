# Variational Autoencoders for Collaborative Filtering

This is my PyTorch implementation of the paper [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara (World Wide Web conference in 2018). 
In particular, I trained a Variational Autoencoder with multinomial likelihood (described in the paper) on the public Movielens-1M dataset. 

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/VAE.png" width="800">

## Requirements
```
PyTorch 1.3
Python 3.6
Numpy
Pandas
SciPy
```

## Scripts
* [DataUtils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/DataUtils.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [Dataset.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Dataset.py): This is the script that defines the Dataset class.
* [Tools.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Tools.py): This is the script that includes the choice of activation functions.
* [Params.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Params.py): This is the script that loads hyper-parameters from a json file.
* [ModelBuilder.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/ModelBuilder.py): This is the script that includes the model building function.
* [Evaluator.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Evaluator.py): This is the script that defines the Evaluator class.
* [Trainer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Trainer.py): This is the script that defines the Trainer class.
* [Logger.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/Logger.py): This is the script that defines the Logger class.
* [BaseModel.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/BaseModel.py): This is the script that defines the Base Model class.
* [MultVAE.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/MultVAE.py): This is the script that defines the Variational Autoencoder with Multinomial Likelihood model class.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/VAE-PyTorch/main.py): This is the main training script.

To reproduce the results, you simply run `python main.py`.

## Results
The model configuration is stored in [the config folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/VAE-PyTorch/config).
The dimension of the encoder module is 200. The dropout rate is set to be 0.5.
The batch size is set to be 512. The learning rate is set to be 0.01.
Adam was used as the optimizer. The data is split to 80% training set and 20% test set.

After training the model for 55 epochs, I got [the best results](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/VAE-PyTorch/saves) on the held-out test set at epoch 5 (thanks to early stopping):
- Precision@100 = 0.0886
- Recall@100 = 0.4115
- NDCG@100 = 0.2508
- Novelty@100 = 2.4775
- Gini Index = 0.2131

I also logged the experiment run on CometML, which can be accessed [here](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/ef7a21a9da5b4fd095ad6a3842decaf8).