# Wide and Deep Learning For Recommender Systems

This is my PyTorch implementation of the paper [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) by Google Inc. (2016).
This model has been productionized and evaluated on [Google Play](https://play.google.com/store?hl=en_US), a commercial mobile app store with over one billion active users and over one million apps.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/pics/Figure1.png" width="800">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/data.py): This is the data processing script.
* [layer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/layer.py): This is the utility script that defines layer classes used in the Wide and Deep model.
* [Wide_Deep.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/Wide_Deep.py): This is the model script that defines the Wide and Deep model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Requirements

```
PyTorch 1.3
Python 3.6
Numpy
Pandas
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch/pics/Figure4.png" width="650">

## Results
I tuned the hyper-parameters of the model according to the paper:
- Number of Dense Embedding Dimensions used in the Deep Component = 16
- Number of Hidden Layers used in the Deep Component = 16
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 2048
- Weight Decay = 0.000001
- Optimizer Method = Adam
- Dropout Rate = 0.2

After being trained for 100 epochs, the model achieves **validation AUC = 0.791** and **test AUC = 0.788**.