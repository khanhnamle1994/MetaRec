# Boltzmann Machines for Collaborative Filtering

This is a series of Boltzmann Machines models for Collaborative Filtering implemented in PyTorch, TensorFlow, and Keras.

Here are the 3 different models:

* [Restricted Boltzmann Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Boltzmann-Machines-Experiments/RBM-CF-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Restricted-Boltzmann-Machines-For-Collaborative-Filtering.pdf))
* [Explainable Restricted Boltzmann Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Boltzmann-Machines-Experiments/Explainable-RBM-CF-TensorFlow) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Explainable-Restricted-Boltzmann-Machines-For-Collaborative-Filtering.pdf))
* [Neural Autoregressive Distribution Estimator](https://github.com/khanhnamle1994/transfer-rec/tree/master/Boltzmann-Machines-Experiments/NADE-CF-Keras) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Neural-Autoregressive-Distribution-Estimator-For-Collaborative-Filtering.pdf))

<img src="https://miro.medium.com/max/1518/1*9Aro2AvQ3V_KmnU_dfEl-A.png" width="1000">

## Download and Process Data
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

## Training Models

To run the Restricted Boltzmann Machines model:

```
python RBM-CF-PyTorch/train.py
```

To run the Explainable Restricted Boltzmann Machines model:

```
python Explainable-RBM-CF-TensorFlow/main.py
```

To run the Neural Autoregressive Distribution Estimator model:

```
python NADE-CF-Keras/run.py
```

## Evaluating Results

Here are the results for all three models after 50 epochs of training:

|      Model      |  RMSE  | Runtime |
|:---------------:|:------:|:-------:|
|       RBM       |  0.590 |  10m56s |
| Explainable RBM | 0.3116 |  1m43s  |
|       NADE      |  0.920 |  90m45s |

* Explainable RBM model has the lowest RMSE and shortest training time.
* NADE, on the other hand, has the highest RMSE and longest training time.