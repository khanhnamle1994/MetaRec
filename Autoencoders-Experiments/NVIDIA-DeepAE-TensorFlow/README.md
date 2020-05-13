# Training Deep AutoEncoders for Collaborative Filtering

This is my TensorFlow implementation of the Deep Autoencoder model in the paper "[Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/abs/1708.01715)" by Oleksii Kuchaiev and Boris Ginsburg at NVIDIA.
This paper continued the [AutoRec](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow) idea to deepen Autoencoder.

- It uses Masked Mean Squared Error as loss function, same as AutoRec.
- It proposes activation functions with non-zero negative part and unbounded positive part works better.
- It uses dropout layers after the latent layer to avoid overfitting.
- It also shows that using large dropout rate after the latent layer allows it to learn robust representations.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/Deep-AutoEncoder.png" width="700">

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
* [data_processor.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/data_processor.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [utils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/utils.py): This is the script that includes utility functions.
* [model.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/model.py): This is the model script that defines the Deep Autoenocder model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/main.py): This is the main training script. You can simply run `python3 main.py` to execute it.

## Results
The full results are stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/NVIDIA-DeepAE-TensorFlow/results).
I tuned the hyperparameters according to the paper:
- Autoencoder architecture layers = [256, 512, 256] (encoder layer of size 256, coding layer of size 512, decoder layer of size 256)
- Activation Function = SELU (scaled exponential linear units)
- Learning Rate = 0.0001
- Batch Size = 256
- L2 Regularizer (Lambda) Value = 0.001
- Optimizer Method = Adam
- Dropout Rate = 0.5
- Number of Training Epochs = 100

After training the model for 100 epochs, I got the **test loss RMSE = 0.8654**.