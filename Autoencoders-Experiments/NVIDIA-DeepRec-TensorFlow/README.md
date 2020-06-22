# Training Deep AutoEncoders for Collaborative Filtering

This is my TensorFlow implementation of the DeepRec model in the paper "[Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/abs/1708.01715)" by Oleksii Kuchaiev and Boris Ginsburg at NVIDIA.
This paper continued the [AutoRec](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow) idea to deepen Autoencoder.

- It uses Masked Root Mean Squared Error as loss function, same as AutoRec.
- It proposes activation functions with non-zero negative part and unbounded positive part works better.
- It uses dropout layers after the latent layer to avoid overfitting.
- It also shows that using large dropout rate after the latent layer allows it to learn robust representations.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/Deep-AutoEncoder.png" width="700">

## Requirements
```
TensorFlow and Keras
Python 3.6
Numpy
Pandas
Matplotlib
Scikit-Learn
```

## Scripts
* [data_processor.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/data_processor.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [utils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/utils.py): This is the script that includes utility functions.
* [model.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/model.py): This is the model script that defines the Deep Autoenocder model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow/results).
I tuned the hyperparameters according to the paper:
- Autoencoder architecture layers = [512, 512, 1024, 512, 512] (encoder layer of size (512, 512, 1024), coding layer of size 1024, decoder layer of size (512, 512, n))
- Activation Function = SELU (scaled exponential linear units)
- Learning Rate = 0.001
- Batch Size = 512
- L2 Regularizer (Lambda) Value = 0.001
- Optimizer Method = Stochastic Gradient Descent
- Momentum = 0.9
- Dropout Rate = 0.8

After training the model for 500 epochs, I got the **Test Masked RMSE = 0.9310**.
I also logged the experiment run on CometML, which can be accessed [here](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/79db0882636549b5bd8a8d66563602f3).