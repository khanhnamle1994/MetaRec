# MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation

This is the PyTorch implementation of the paper "[MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation](https://arxiv.org/abs/1908.00413)" that is adapted from the [original codebase](https://github.com/hoyeoplee/MeLU).
MeLU is a meta-learning-based system that alleviates the cold-start problem by estimating user preferences based on only a small number of items.
In addition, MeLU uses an evidence candidate selection strategy that determines distinguishing items for customized preference estimation.

## Scripts
* [data_loader.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/data_loader.py): This is the data loader script.
* [data_generator.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/data_generator.py): This is the data generator script.
* [config.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/config.py): This is the configuration script that includes hyper-parameters used to train MeLU.
* [embeddings.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/embeddings.py): This is the embedding script that converts used and item inputs into user and item embeddings.
* [MeLU.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/MeLU.py): This is the model script that defines MeLU.
* [train.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/train.py): This is the training script that trains MeLU.
* [evidence_candidate.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/evidence_candidate.py): This is the candidate generation script that returns ranked recommended items.
* [main.py](https://github.com/khanhnamle1994/transfer-rec/blob/master/Meta-Learning/MeLU/main.py): This is the main script that executes the whole code.

## Requirements

```
pytorch 1.3
python 3.6
tqdm 4.32
pandas 0.24
```

## Citation

```
@inproceedings{lee2019melu,
  title={MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation},
  author={Lee, Hoyeop and Im, Jinbae and Jang, Seongwon and Cho, Hyunsouk and Chung, Sehee},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1073--1082},
  year={2019},
  organization={ACM}
}
```