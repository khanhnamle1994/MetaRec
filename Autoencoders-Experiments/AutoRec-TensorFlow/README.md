# Autoencoders Meet Collaborative Filtering

This is my TensorFlow implementation of the AutoRec model in the paper "[Autoencoders Meet Collaborative Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)" by Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie.
This is one of the first papers that proposes an autoencoder framework for collaborative filtering.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-TensorFlow/AutoRec.png" width="700">

## Requirements
```
TensorFlow 1.4
Python 3.6
Numpy
Pandas
```

## Scripts
* [data_preprocessor.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-TensorFlow/data_preprocessor.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [AutoRec.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-TensorFlow/AutoRec.py): This is the model script that defines the AutoRec model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-TensorFlow/main.py): This is the main training script. You can simply run `python main.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow/results).
I tuned the hyper-parameters according to the paper:
- Number of Hidden Units = 500 (within the Autoencoder's Hidden Layer)
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 512
- Decay the Learning Rate for every 50 epochs
- L2 Regularizer (Lambda) Value = 1
- Optimizer Method = Adam
- Random Seed = 1994

After training the model for 500 epochs, I got the **test loss RMSE = 0.910**.
I also logged the experiment run on CometML, which can be accessed [here](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/6c606d10c5f24628a88865c0270361ec).