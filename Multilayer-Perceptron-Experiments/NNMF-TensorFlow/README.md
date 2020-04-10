# Neural Network Matrix Factorization

This is my TensorFlow implementation of the paper [Neural Network Matrix Factorization](https://arxiv.org/abs/1511.06443) by Gintare Dziugaite and Daniel M. Roy (2015).
This is one of the first papers that proposes a Multi-Layer Perceptron framework for collaborative filtering.

## Requirements
```
TensorFlow 1.4
Python 3.6
Numpy
Pandas
```

## Scripts
* [core.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/NNMF-TensorFlow/core.py): This is the core engine script that defines the training, evaluating, and testing functions.
* [model.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/NNMF-TensorFlow/model.py): This is the model script that defines the NNMF model.
* [run.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/NNMF-TensorFlow/run.py): This is the main execution script. You can simply run `python run.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/NNMF-TensorFlow/results).
I tuned the hyper-parameters of the Multi-Layer Perceptron according to the paper:
- Number of Hidden Units = 50
- Number of Hidden Layers = 6
- Number of Latent Dimensions (D) = 10
- Number of Second-Order Latent Dimensions (D_Prime) = 65
- Number of Vector Dimensions (K) = 48
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 128
- L2 Regularizer (Lambda) Value = 54
- Optimizer Method = RMSProp
- Dropout Rate = 0.05

After training the model for 1000 epochs and specifying early stopping, I got the lowest loss after 652 epochs with **validation RMSE = 1.227** and **test RMSE = 1.229**.