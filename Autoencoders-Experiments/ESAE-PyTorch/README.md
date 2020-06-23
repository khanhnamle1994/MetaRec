# Embarrassingly Shallow Autoencoders for Sparse Data

This is a PyTorch implementation of the ESAE model in the paper "[Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/abs/1905.03375)" by Harald Steck published in the proceedings of ’The Web Conference’ (WWW 2019).
As observed below, the self-similarity of each item is constrained to 0 between the input and output layers. The Python code of the learning algorithm is given in Algorithm 1 in the paper.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/ESAE.jpg" width="750">

## Requirements
```
PyTorch 1.3
Python 3.6
Numpy
Pandas
SciPy
```

## Scripts
* [DataUtils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/DataUtils.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [Dataset.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Dataset.py): This is the script that defines the Dataset class.
* [Tools.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Tools.py): This is the script that includes the choice of activation functions.
* [Params.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Params.py): This is the script that loads hyper-parameters from a json file.
* [ModelBuilder.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/ModelBuilder.py): This is the script that includes the model building function.
* [Evaluator.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Evaluator.py): This is the script that defines the Evaluator class.
* [Trainer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Trainer.py): This is the script that defines the Trainer class.
* [Logger.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/Logger.py): This is the script that defines the Logger class.
* [BaseModel.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/BaseModel.py): This is the script that defines the Base Model class.
* [ESAE.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/ESAE.py): This is the script that defines the Embarrassingly Shallow Autoencoder Model class.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/ESAE-PyTorch/main.py): This is the main training script.

To reproduce the results, you simply run `python main.py`.

## Results
The model configuration is stored in [the config folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/ESAE-PyTorch/config).
ESAE is a linear model without a hidden layer. The binary input vector indicates which items a user has interacted with, and the model's objective (in its output layer) is to predict the best items to recommend to the user.
The author derive the closed-form solution of its convex training objective.
The model is trained with Adam optimizer, a batch size of 512 and learning rate of 0.01. The data is split to 80% training set and 20% test set.

The [results](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/ESAE-PyTorch/saves) on the held-out test set after 500 epochs of training are:
- Precision@100 = 0.0757
- Recall@100 = 0.4181
- NDCG@100 = 0.2561
- Novelty@100 = 2.4604
- Gini Index = 0.2379

I also logged the experiment run on CometML, which can be accessed [here](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/237651ecb58545069ef176194b2d3935).