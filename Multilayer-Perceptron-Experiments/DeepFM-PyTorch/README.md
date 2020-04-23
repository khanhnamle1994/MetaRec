# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

This is my PyTorch implementation of the paper [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf) by Huifeng Guo et. al (2017).
This approach is an extension of Google's "[Wide and Deep Model](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/Wide-and-Deep-PyTorch)".
More specifically, DeepFM has a shared input to its “wide” and “deep” parts, with no need of feature engineering besides raw features.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch/pics/Figure1.png" width="800">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch/data.py): This is the data processing script.
* [layer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch/layer.py): This is the utility script that defines layer classes used in the DeepFM model.
* [DeepFM.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch/DeepFM.py): This is the model script that defines the DeepFM model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Requirements

```
PyTorch 1.3
Python 3.6
Numpy
Pandas
```

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

After being trained for 100 epochs, the model achieves **validation AUC = 0.7918** and **test AUC = 0.7915** with **runtime = 1h 10m 50s**.
The results can be viewed at [this Weights & Biases link](https://app.wandb.ai/khanhnamle1994/multi_layer_perceptron_collaborative_filtering/runs/pnu0yndp).