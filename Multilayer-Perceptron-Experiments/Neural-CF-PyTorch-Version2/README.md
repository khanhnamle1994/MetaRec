# Neural Collaborative Filtering (Version 2)

This is my PyTorch implementation of the paper [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) by Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017).
There is also a [Version 1](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version1) implementation, which includes all 3 collaborative filtering models in the paper: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF).
In this implementation, I only implemented the NMF module. Furthermore, while in version 1, the evaluation metrics are Hit Ratio and Normalized Discounted Cumulative Gain; in this version, the evaluation metric is AUC.
This is because the rating data is transformed into binary targets, in which rating <= 3 is class 0 and rating > 3 is class 1.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/pics/Fig2.png" width="850">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/data.py): This is the data processing script.
* [layer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/layer.py): This is the utility script that defines layer classes used in the Neural CF model.
* [NeuralCF.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/NeuralCF.py): This is the model script that defines the Neural CF model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Requirements

```
PyTorch 1.3
Python 3.6
Numpy
Pandas
```

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-CF-PyTorch-Version2/pics/Fig3.png" width="700">

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

## Results
Here are the model hyper-parameters chosen:
- Number of Dense Embedding Dimensions used in the Deep Component = 16
- Number of Hidden Layers used in the Deep Component = 16
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 512
- Weight Decay = 0.000001
- Optimizer Method = Adam
- Dropout Rate = 0.5

After being trained for 100 epochs, the model achieves **validation AUC = 0.7673** and **test AUC = 0.7688** with **runtime = 54m 15s**.
The results can be viewed at [this Weights & Biases link](https://app.wandb.ai/khanhnamle1994/multi_layer_perceptron_collaborative_filtering/runs/jmx7q24t).