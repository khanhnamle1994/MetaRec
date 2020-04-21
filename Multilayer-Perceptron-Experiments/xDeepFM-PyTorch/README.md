# xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

This is my PyTorch implementation of the paper [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) by Jianxun Lian et. al (2018).
This approach is an extension of Huifeng Guo et. al's "[Deep Factorization Machine](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch)".
More specifically, xDeepFM jointly learns explicit and implicit high-order feature interactions effectively and requires no manual feature engineering.
It accomplishes this with a Compressed Interaction Network module that learns high-order feature interactions explicitly.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/pics/Figure4.png" width="850">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/data.py): This is the data processing script.
* [layer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/layer.py): This is the utility script that defines layer classes used in the xDeepFM model.
* [xDeepFM.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/xDeepFM.py): This is the model script that defines the xDeepFM model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Requirements

```
PyTorch 1.3
Python 3.6
Numpy
Pandas
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/xDeepFM-PyTorch/pics/Figure5.png" width="700">

## Results
Here are the model hyper-parameters chosen:
- Number of Dense Embedding Dimensions used in the Deep Component = 16
- Number of Hidden Layers used in the Deep Component = 16
- Compressed Interaction network with size (16, 16)
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 512
- Weight Decay = 0.000001
- Optimizer Method = Adam
- Dropout Rate = 0.5

After being trained for 100 epochs, the model achieves **validation AUC = 0.7408** and **test AUC = 0.7429** with **runtime = 2h 15m 17s**.