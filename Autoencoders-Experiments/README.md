# Autoencoders for Collaborative Filtering

This is a series of Autoencoders models for Collaborative Filtering implemented in PyTorch and TensorFlow.

Here are the 6 different models:

* [AutoRec](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/AutoRec-TensorFlow) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/AutoRec-Autoencoders-Meet-Collaborative-Filtering.pdf))
* [DeepRec](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/NVIDIA-DeepRec-TensorFlow) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Training-Deep-Autoencoders-For-Collaborative-Filtering.pdf))
* [Collaborative Denoising Autoencoders](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/CDAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Collaborative-Denoising-Autoencoders-for-TopN-Recommendation-System.pdf))
* [Multinomial Variational Autoencoders](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/VAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Variational-Autoencoders-for-Collaborative-Filtering.pdf))
* [Sequential Variational Autoencoders](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/SVAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Sequential-Variational-Autoencoders-for-Collaborative-Filtering.pdf))
* [Embarrassingly Shallow Autoencoders](https://github.com/khanhnamle1994/MetaRec/tree/master/Autoencoders-Experiments/ESAE-PyTorch) ([paper](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/Embarrassingly-Shallow-Autoencoders-for-Sparse-Data.pdf))

<img src="https://s3.ca-central-1.amazonaws.com/assets.jmir.org/assets/72cb8a4949fe35aa88374a70749f6ec2.png" width="1000">

## Dataset
You can download the MovieLens-1M dataset from [this folder](https://github.com/khanhnamle1994/MetaRec/tree/master/ml-1m).

## Training Models

To run the AutoRec model:

```
python AutoRec-TensorFlow/main.py
```

To run the DeepRec model:

```
python NVIDIA-DeepRec-TensorFlow/main.py
```

To run the Collaborative Denoising Autoencoders model:

```
python CDAE-PyTorch/main.py
```

To run the Multinomial Variational Autoencoders model:

```
python VAE-PyTorch/main.py
```

To run the Sequential Variational Autoencoders model:

```
python SVAE-PyTorch/main.py
```

To run the Embarrassingly Shallow Autoencoders model:

```
python ESAE-PyTorch/main.py
```

## Evaluating Results

Here are the results for all these models after training:

|  Model  | Epochs |  RMSE  | Precision@100 | Recall@100 | NDCG@100 |  Runtime |
|:-------:|:------:|:------:|:-------------:|:----------:|:--------:|:--------:|
| AutoRec |   500  |  0.910 |               |            |          |  35m16s  |
| DeepRec |   500  | 0.9310 |               |            |          |  54m24s  |
|   CDAE  |   141  |        |     0.0894    |   0.4137   |  0.2528  |  17m29s  |
| MultVAE |   55   |        |     0.0886    |   0.4115   |  0.2508  |   6m31s  |
|   SVAE  |   50   |        |     0.0818    |   0.5850   |  0.3807  | 6h37m19s |
|   ESAE  |   50   |        |     0.0757    |   0.4181   |  0.2561  |  10m12s  |

- For rating prediction:
    - AutoRec performs better than DeepRec: lower RMSE and shorter runtime.
    - This is quite surprising, as DeepRec is a deeper architecture than AutoRec
- For ranking predicton:
    - The SVAE model has the best result; however, it also takes order of magnitude longer to train.
    - Between the remaining three models: CDAE has the highest Precision@100, ESAE has the highest Recall@100 and NDCG@100, and MultVAE has the shortest runtime.

The results are captured in [CometML](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/) - which is is a fantastic tool that keeps track of model experiments and logs all necessary metrics in a single dashboard.