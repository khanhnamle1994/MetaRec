# MetaHIN: Meta-Learning on Heterogeneous Information Networks for Cold-start Recommendation

This is the PyTorch implementation of the paper "[Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation](https://yuanfulu.github.io/publication/KDD-MetaHIN.pdf)" that is adapted from the [original codebase](https://github.com/rootlu/MetaHIN).
MetaHIN is a novel attempt to exploit meta-learning on Heterogeneous Information Networks for cold-start recommendation, which alleviates the cold-start problem at both data and model levels.
It leverages multi-faceted semantic contexts and a co-adaption meta-learner in order to learn finer-grained semantic priors for new tasks in both semantic and task-wise manners.

## Scripts
* [data_helper.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/data_helper.py): This is the data loader script.
* [data_processor.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/data_processor.py): This is the data processor script.
* [config.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/config.py): This is the configuration script that includes hyper-parameters used to train MetaHIN.
* [embedding_init.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/embedding_init.py): This is the embedding script that converts user and item input features into user and item embeddings.
* [metaHIN.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/metaHIN.py): This is the model script that defines MetaHIN.
* [meta_learner.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/meta_learner.py): This is the training script that trains MAMO by updating the parameters in a meta-learning paradigm.
* [evaluation.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/evaluation.py): This is the evaluation script that evaluates the performance of learned embeddings w.r.t clustering and classification.
* [main.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MetaHIN/main.py): This is the main script that executes the whole code.

## Requirements

```
- Python 3.6.9
- PyTorch 1.4.0
```
See the detailed [requirements](https://github.com/rootlu/MetaHIN/blob/master/requirements.txt).

## Citation

```
@inproceedings{lu2020meta,
  title={Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation},
  author={Lu, Yuanfu and Fang, Yuan and Shi, Chuan},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1563--1573},
  year={2020}
}
```