# Explainable Restricted Boltzmann Machines for Collaborative Filtering

This is my TensorFlow implementation of the paper [Explainable Restricted Boltzmann Machines for Collaborative Filtering](https://arxiv.org/abs/1606.07129) by Behnoush Abdollahi and Olfa Nasraoui (2016).
In particular, I trained the RBM model with contrastive divergence on the public Movielens-1M dataset, and then calculated the **explainability score** for each recommendation that is based on its neighbors.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Explainable-RBM-CF-TensorFlow/pics/Explainable-RBM.png" width="650">

## Requirements
```
TensorFlow 1.3
Python 3.6
Numpy
Pandas
Matplotlib
```

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).

## Scripts
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Explainable-RBM-CF-TensorFlow/main.py): This is the main training script. You can simply run `python main.py` to execute it.

## Results
I obtained a reconstruction RMSE (Root Mean Squared Error) value of 0.3116 after training the model for 50 epochs. Training time takes 1m43s.
The full run is logged into Weights and Biases [here](https://app.wandb.ai/khanhnamle1994/boltzmann_machines_collaborative_filtering/runs/wf3pescx).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Explainable-RBM-CF-TensorFlow/pics/result.png" width="600">

## Sample Recommendation List
This is a sample recommendation list for a mock user with explainable scores attached each movie.

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Boltzmann-Machines-Experiments/Explainable-RBM-CF-TensorFlow/pics/Recommendations-Example.png" width="1000">