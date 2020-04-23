# Matrix Factorization for Collaborative Filtering

This is a series of Matrix Factorization models for Collaborative Filtering implemented in PyTorch. The code is heavily inspired by [Chris Moody's tutorial on Deep Recommendations in PyTorch](https://docs.google.com/presentation/d/1gv7osHoSX8CHf0uzKSqOlxmmAvPPdmstL0nrZHWiHQM/edit#slide=id.p).

Here are the 7 different models:

* [Vanilla Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Vanilla-MF)
* [Matrix Factorization with Biases](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Biases)
* [Matrix Factorization with Side Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Side-Features)
* [Matrix Factorization with Temporal Features](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Temporal-Features)
* [Factorization Machines](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Factorization-Machines)
* [Matrix Factorization with Mixture of Tastes](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/MF-Mixture-Tastes)
* [Variational Matrix Factorization](https://github.com/khanhnamle1994/transfer-rec/tree/master/Matrix-Factorization-Experiments/Variational-MF)

An accompanied Medium blog post has been written up and can be viewed here: [The 7 Variants of Matrix Factorization For Collaborative Filtering](https://towardsdatascience.com/recsys-series-part-4-the-7-variants-of-matrix-factorization-for-collaborative-filtering-368754e4fab5)

<img src="https://miro.medium.com/max/4800/1*b4M7o7W8bfRRxdMxtFoVBQ.png" width="1000">

## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Download and Process Data
* You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/data.py) is the script that pre-processes the data.

## Training Models

To run the Vanilla Matrix Factorization model:

```
python Vanilla-MF/train.py
```

To run the Matrix Factorization with Biases model:

```
python MF-Biases/train.py
```

To run the Matrix Factorization with Side Features model:

```
python MF-Side-Features/train.py
```

To run the Matrix Factorization with Temporal Features model:

```
python MF-Temporal-Features/train.py
```

To run the Factorization Machines model:

```
python Factorization-Machines/train.py
```

To run the Matrix Factorization with Mixture of Tastes model:

```
python MF-Mixture-Tastes/train.py
```

To run the Variational Matrix Factorization model:

```
python Variational-MF/train.py
```

## Evaluating Results

Here are the results for all these models after 50 epochs of training:

|         Model         | Training Loss |   Test Loss   | Training Time |
|:---------------------:|:-------------:|:-------------:|:-------------:|
|       Vanilla MF      |     0.6947    |     0.8174    |     6m5s      |
|       MF Biases       |     0.6789    |     0.7895    |     11m38s    |
|    MF Side Features   |     0.6602    |     0.7843    |     13m34s    |
|  MF Temporal Features |     0.7088    |     0.7939    |     18m51s    |
| Factorization Machine |     0.6542    |     0.8225    |     3m40s     |
|  MF Mixture of Tastes |     0.6366    |     0.7878    |     13m44s    |
|     Variational MF    |     0.6206    |     0.8385    |     16m51s    |

* Variational MF has the lowest training loss
* MF with Side Features has the lowest test loss
* Factorization Machines has the lowest training time