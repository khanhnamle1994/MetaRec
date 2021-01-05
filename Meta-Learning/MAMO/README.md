# MAMO: Memory-Augmented Meta-Optimization for Cold-Start Recommendation

This is the PyTorch implementation of the paper "[MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation](https://arxiv.org/abs/2007.03183)" that is adapted from the [original codebase](https://github.com/dongmanqing/Code-for-MAMO).
MAMO includes two memory matrices that can store task-specific memories and feature-specific memories.
Specifically, the feature-specific memories are used to guide the model with personalized parameter initialization, while the task-specific memories are used to guide the model fast predicting the user preference.
In addition, MAMO is optimized with a meta-optimization approach.

## Scripts
* Inside [data_prep](https://github.com/khanhnamle1994/MetaRec/tree/master/Meta-Learning/MAMO/data_prep):
    * [prepare_data.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/data_prep/prepare_data.py): This is the data generator script.
    * [prepare_list.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/data_prep/prepare_list.py): This is the utility script that converts user and item data into Python lists.
    * [preprocess_MovieLens.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/data_prep/preprocess_MovieLens.py): This is the MovieLens data pre-processing script.
* Inside [modules](https://github.com/khanhnamle1994/MetaRec/tree/master/Meta-Learning/MAMO/modules):
    * [info_embedding.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/modules/info_embedding.py): This is the embedding script that generates user and item embeddings.
    * [input_loading.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/modules/input_loading.py): This is the loading script that loads user and item inputs.
    * [memories.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/modules/memories.py): This is the memories script that create feature-specific and task-specific memory components.
    * [rec_model.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/modules/rec_model.py): This is the model script that builds the underlying recommendation model.
* [configs.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/configs.py): This is the configuration script that includes hyper-parameters used to train MAMO.
* [models.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/models.py): This is the model script that initializes the base model class and local update procedure.
* [mamoRec.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/mamoRec.py): This is the main script that executes the MAMO.
* [utils.py](https://github.com/khanhnamle1994/MetaRec/blob/master/Meta-Learning/MAMO/utils.py): This is the utility script that contains variouns functions to support training and evaluation.

## Requirements

```
pytorch 1.3
python 3.6
tqdm 4.32
pandas 0.24
```

## Citation

```
@inproceedings{dong2020mamo,
  title={MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation},
  author={Manqing, Dong and Feng, Yuan and Lina, Yao and Xiwei, Xu and Liming, Zhu},
  booktitle={26th SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2020}
}
```