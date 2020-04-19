# Variational Autoencoders for Collaborative Filtering (Version 2)

This is my PyTorch implementation of the paper [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814) by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara (2018). 
In particular, I trained a Denoising Autoencoder and a Variational Autoencoder with multinomial likelihood (described in the paper) on the public Movielens-1M and Movielens-20M datasets. 
Every data preprocessing step and code follows exactly from the [authors' Repo](https://github.com/dawenl/vae_cf).
This is an extension from the [Version 1](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/VAE-CF-PyTorch-Version1) repo, where I only worked with the Movielens-20M dataset.

## Requirements

```
PyTorch 1.3 & Python 3.6
Numpy
Pandas
```

## Usage

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

On running `main.py`, it asks you whether to train on MovieLens-1m or MovieLens-20m. (Enter 1 or 20)

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## DAE

```bash
python main.py --template train_dae
```

## VAE

Search for the optimal beta

```bash
python main.py --template train_vae_search_beta
```

Then fill out the optimal beta value in `templates.py`. Then, run the following.

``` bash
python main.py --template train_vae_give_beta
```

# Test Set Results

Here are the results for all these models after 100 epochs of training:
- **Multi-DAE-0** is the Denoising Autoencoder model with no hidden layer.
- **Multi-DAE-1** is the Denoising Autoencoder model with one hidden layer.
- **Multi-DAE-2** is the Denoising Autoencoder model with two hidden layers.
- **Multi-VAE-0** is the Variational Autoencoder model with no hidden layer.
- **Multi-VAE-1** is the Variational Autoencoder model with one hidden layer.
- **Multi-VAE-2** is the Variational Autoencoder model with two hidden layers.

## MovieLens-1m

|   Metric   | Multi-DAE-0 | Multi-DAE-1 | Multi-DAE-2 | Multi-VAE-0 | Multi-VAE-1 | Multi-VAE-2 |
|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|  Recall@1  |    0.5152   |    0.4805   |    0.4526   |    **0.5449**   |    0.5043   |    0.4264   |
|  Recall@5  |    **0.4319**   |    0.4048   |    0.3885   |    0.4202   |    0.4217   |    0.3872   |
|  Recall@10 |    **0.3974**   |    0.3797   |    0.3684   |    0.3743   |    0.4217   |    0.3619   |
|  Recall@20 |    **0.3942**   |    0.3891   |    0.3785   |    0.3713   |    0.3800   |    0.3768   |
|  Recall@50 |    **0.4821**   |    0.4814   |    0.4753   |    0.4480   |    0.4685   |    0.4798   |
| Recall@100 |    0.5920   |    **0.5978**   |    0.5878   |    0.5616   |    0.5837   |    0.5871   |
|   NDCG@1   |    0.5152   |    0.4805   |    0.4526   |    **0.5449**   |    0.5043   |    0.4264   |
|   NDCG@5   |    **0.4507**   |    0.4224   |    0.4033   |    0.4476   |    0.4407   |    0.3953   |
|   NDCG@10  |    **0.4186**   |    0.3980   |    0.3801   |    0.4049   |    0.4050   |    0.3730   |
|   NDCG@20  |    **0.4024**   |    0.3903   |    0.3746   |    0.3849   |    0.3870   |    0.3668   |
|   NDCG@50  |    **0.4263**   |    0.4187   |    0.4074   |    0.4020   |    0.4128   |    0.4055   |
|  NDCG@100  |    **0.4681**   |    0.4634   |    0.4522   |    0.4439   |    0.4560   |    0.4491   |

## MovieLens-20m

|   Metric   | Multi-DAE-0 | Multi-DAE-1 | Multi-DAE-2 | Multi-VAE-0 | Multi-VAE-1 | Multi-VAE-2 |
|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|  Recall@1  |    **0.3776**   |    0.3605   |    0.3460   |    0.3100   |    0.3326   |    0.3185   |
|  Recall@5  |    **0.3108**   |    0.3014   |    0.2957   |    0.2496   |    0.2756   |    0.2675   |
|  Recall@10 |    **0.3259**   |    0.3247   |    0.3187   |    0.2605   |    0.2938   |    0.2876   |
|  Recall@20 |    0.3874   |    **0.3900**   |    0.3857   |    0.3117   |    0.2938   |    0.3464   |
|  Recall@50 |    0.5212   |    **0.5312**   |    0.5303   |    0.4298   |    0.3505   |    0.4822   |
| Recall@100 |    0.6433   |    **0.6551**   |    0.6548   |    0.5413   |    0.6077   |    0.6050   |
|   NDCG@1   |    **0.3776**   |    0.3605   |    0.3460   |    0.3100   |    0.3326   |    0.3185   |
|   NDCG@5   |    0.3200   |    **0.4083**   |    0.3000   |    0.2586   |    0.2831   |    0.2730   |
|   NDCG@10  |    **0.3148**   |    0.3095   |    0.3015   |    0.2533   |    0.2918   |    0.2731   |
|   NDCG@20  |    **0.3300**   |    0.3284   |    0.3222   |    0.2652   |    0.2967   |    0.2895   |
|   NDCG@50  |    0.3750   |    **0.3778**   |    0.3732   |    0.3038   |    0.2967   |    0.3359   |
|  NDCG@100  |    0.4167   |    **0.4202**   |    0.4164   |    0.3418   |    0.3838   |    0.3778   |

=> From the Recall and NDCG results above, the Denoising Autoencoder model with no or one hidden layer performs best in the majority of the case.