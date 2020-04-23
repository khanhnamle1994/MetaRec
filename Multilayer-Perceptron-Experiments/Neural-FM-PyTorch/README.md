# Neural Factorization Machines For Sparse Predictive Analytics

This is my PyTorch implementation of the paper [Neural Factorization Machines For Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf) by Xiangnan He and Tat-Seng Chua (2017).
This approach is very similar to Huifeng Guo et. al's "[Deep Factorization Machine](https://github.com/khanhnamle1994/transfer-rec/tree/master/Multilayer-Perceptron-Experiments/DeepFM-PyTorch)".
More specifically, the model seamlessly combines the linearity of Factorization Machine in modeling second-order feature interactions and the non-linearity of neural network in modeling higher-order feature interactions.
There is also a [TensorFlow implementation](https://github.com/hexiangnan/neural_factorization_machine) from the authors themselves.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch/pics/Figure2.png" width="850">

## Scripts
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch/data.py): This is the data processing script.
* [layer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch/layer.py): This is the utility script that defines layer classes used in the Neural FM model.
* [NeuralFM.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch/NeuralFM.py): This is the model script that defines the Neural FM model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Multilayer-Perceptron-Experiments/Neural-FM-PyTorch/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

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
- Number of Dense Embedding Dimensions used in the Deep Component = 64
- Number of Hidden Layers used in the Deep Component = 64
- Activation Function = Sigmoid
- Learning Rate = 0.001
- Batch Size = 512
- Weight Decay = 0.000001
- Optimizer Method = Adam
- Dropout Rate = 0.2

After being trained for 100 epochs, the model achieves **validation AUC = 0.7560** and **test AUC = 0.7589** with **runtime = 1h 36m 0s**.
The results can be viewed at [this Weights & Biases link](https://app.wandb.ai/khanhnamle1994/multi_layer_perceptron_collaborative_filtering/runs/5tsd38zl).