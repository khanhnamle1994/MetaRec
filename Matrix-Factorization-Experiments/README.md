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

## Requirements
```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
Pytorch-Ignite
Sklearn
TensorboardX
```

## Processing and Loading Data
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [data.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/data.py): This is the script that pre-processes the data.
* [loader.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Matrix-Factorization-Experiments/loader.py): This is the script that loads the data.

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

|         Model         | Training Loss | Test Accuracy | Training Time |
|:---------------------:|:-------------:|:-------------:|:-------------:|
|       Vanilla MF      |     0.7068    |     0.8174    |     8m22s     |
|       MF Biases       |     0.6623    |     0.788     |     12m32s    |
|    MF Side Features   |     0.7248    |     0.7893    |     13m51s    |
|  MF Temporal Features |     0.7464    |     0.7954    |     21m23s    |
| Factorization Machine |     0.7338    |     0.8208    |     3m11s     |
|  MF Mixture of Tastes |     0.6638    |     0.7871    |     14m35s    |
|     Variational MF    |     0.6929    |     0.8305    |     17m42s    |
