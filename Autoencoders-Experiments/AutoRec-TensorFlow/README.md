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
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow/results). I tuned the hyperparameters according to the paper:
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 128
- Decay the Learning Rate for each 50 epochs
- Number of Hidden Neurons = 500 (within the Autoencoder)
- L2 Regularizer (Lambda) Value = 1
- Optimizer Method = Adam
- Random Seed = 1000
- Number of Training Epochs = 100

After training the model for 100 epochs, I got the **test loss RMSE = 0.9102**.
