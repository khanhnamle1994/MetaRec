# Collaborative Denoising Autoencoders for Top-N Recommender Systems

This is a PyTorch implementation of the CDAE model in the paper "[Collaborative Denoising Autoencoders for Top-N Recommender Systems](https://alicezheng.org/papers/wsdm16-cdae.pdf)" by Yao Wu, Christopher DuBois, Alice Zheng, and Martin Ester published at the 9th ACM International Conference on Web Search and Data Mining (WSDM '16).
The code is adapted from the C++ code provided by the author's [original repository](https://github.com/jasonyaw/CDAE).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/CDAE.png" width="800">

## Requirements
```
PyTorch 1.3
Python 3.6
Numpy
Pandas
SciPy
```

## Scripts
* [DataUtils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/DataUtils.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [Dataset.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Dataset.py): This is the script that defines the Dataset class.
* [Tools.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Tools.py): This is the script that includes the choice of activation functions.
* [Params.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Params.py): This is the script that loads hyper-parameters from a json file.
* [ModelBuilder.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/ModelBuilder.py): This is the script that includes the model building function.
* [Evaluator.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Evaluator.py): This is the script that defines the Evaluator class.
* [Trainer.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Trainer.py): This is the script that defines the Trainer class.
* [Logger.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/Logger.py): This is the script that defines the Logger class.
* [BaseModel.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/BaseModel.py): This is the script that defines the Base Model class.
* [DAE.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/DAE.py): This is the script that defines the Denoising Autoencoder Model class.
* [CDAE.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/CDAE.py): This is the script that defines the Collaborative Denoising Autoencoder Model class.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/CDAE-PyTorch/main.py): This is the main training script.

To reproduce the results, you simply run `python main.py`.

## Results
The model hyper-parameters are stored in [the config folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/CDAE-PyTorch/config).
According to the paper, the CDAE architecture includes 50 hidden layers. The corruption ratio for the data to the input layer is set to be 0.5.
TanH was used to optimize the loss function coupled with a batch size of 1024 and learning rate of 0.01. The data is split to 80% training set and 20% test set.

After training the model for 500 epochs, I got [the best results](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/CDAE-PyTorch/saves) on the held-out test set at epoch 145:
- Precision@100 = 0.0899
- Recall@100 = 0.4126
- NDCG@100 = 0.2522