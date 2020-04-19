import numpy as np
import pandas as pd
import torch.utils.data


class MovieLens1MDataset(torch.utils.data.Dataset):
    """
        MovieLens 1M Dataset
        Data preparation: treat samples with a rating less than 3 as negative samples
        :param dataset_path: MovieLens dataset path
        Reference: https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep='::', engine='python', header=None):
        # Read the data into a Pandas dataframe
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]

        # Retrieve the items and ratings data
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)

        # Get the range of the items
        self.field_dims = np.max(self.items, axis=0) + 1

        # Initialize NumPy arrays to store user and item indices
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        """
        :return: number of total ratings
        """
        return self.targets.shape[0]

    def __getitem__(self, index):
        """
        :param index: current index
        :return: the items and ratings at current index
        """
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        """
        Preprocess the ratings into negative and positive samples
        :param target: ratings
        :return: binary ratings (0 or 1)
        """
        target[target <= 3] = 0  # ratings less than or equal to 3 classified as 0
        target[target > 3] = 1  # ratings bigger than 3 classified as 1
        return target
