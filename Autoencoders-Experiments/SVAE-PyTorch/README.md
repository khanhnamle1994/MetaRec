# Sequential Variational Autoencoders for Collaborative Filtering

This is a PyTorch implementation of the SVAE model in the paper "[Sequential Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1811.09975)" by Noveen Sachdeva, Giuseppe Manco, Ettore Ritacco, and Vikram Pudi published at the 12th ACM International Conference on Web Search and Data Mining (WSDM '19).
This paper extends the work by [Dawen Liang et al.](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/VAE-CF-PyTorch-Version1) by incorporating recurrent neural networks into the framework to address the temporal nature of the data.
The code is adapted from the notebook provided by the author's [original repository](https://github.com/noveens/svae_cf).

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/SVAE.png" width="800">

## Requirements
```
PyTorch 1.3
Python 3.6
Numpy
Pandas
Matplotlib
```

## Scripts
* [data_preprocessor.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/data_processor.py): This is the script that loads and pre-processes the [MovieLens1M dataset](https://github.com/khanhnamle1994/transfer-rec/tree/master/ml-1m).
* [data_parser.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/data_parser.py): This is the script that parses the processed data to use for training.
* [utils.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/utils.py): This is the script that includes utility functions.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/train.py): This is the script that defines the training function.
* [evaluate.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/evaluate.py): This is the script that defines the evaluation function.
* [model.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/model.py): This is the script that defines the SVAE model.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/main.py): This is the main training script.

To reproduce the results, first you run `python3 data_preprocessor.py` and then run `python3 main.py`.

## Results
The processed data is stored in [this folder](https://github.com/khanhnamle1994/transfer-rec/tree/master/Autoencoders-Experiments/SVAE-PyTorch/processed_data).
According to the paper, the SVAE architecture includes an embedding layer of size 256, a recurrent layer realized as a GRU with 200 cells, and two encoding layers (of size 150 and 64) and finally two decoding layers (again, of size 64 and 150).
The number K of latent factors for the VAE is set to be 64. Adam was used to optimize the loss function coupled with a weight decay of 0.01 and batch size of 1 (because we don't pack multiple sequences in the same batch).

After training the model for 50 epochs, I got [these results](https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/saved_logs/svae_ml1m_log_optimizer_adam_weight_decay_0.005_loss_type_next_k_item_embed_size_256_rnn_size_200_latent_size_64.txt) on the held-out set:
- NDCG@10 = 25.7944 and NDCG@100 = 38.0714
- Rec@10 = 18.9233 and Rec@100 = 58.4987
- Prec@10 = 20.2 and Prec@100 = 8.18

I also logged the experiment run on CometML, which can be accessed [here](https://www.comet.ml/khanhnamle1994/autoencoders-movielens1m/e9a6c227376149d4a9a61f54516a353e)

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/saved_plots/learning_curve_svae_ml1m.png" width="800">

<img src="https://github.com/khanhnamle1994/transfer-rec/blob/master/Autoencoders-Experiments/SVAE-PyTorch/saved_plots/seq_len_vs_ndcg_SVAE_ML1M.png" width="800">