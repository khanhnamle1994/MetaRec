# Import Libraries
import torch
from sklearn.utils import shuffle


# Initialize a Loader class
class Loader():
    # Set the iterator
    current = 0

    def __init__(self, x, y, batchsize=1024, do_shuffle=True):
        """
        :param x: features
        :param y: target
        :param batchsize: batch size = 1024
        :param do_shuffle: shuffle mode turned on
        """
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.batches = range(0, len(self.y), batchsize)
        if do_shuffle:
            # Every epoch re-shuffle the dataset
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        # Reset & return a new iterator
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        # Return the number of batches
        return int(len(self.x) / self.batchsize)

    def __next__(self):
        # Update iterator and stop iteration until the batch size is out of range
        n = self.batchsize
        if self.current + n >= len(self.y):
            raise StopIteration
        i = self.current

        # Transform NumPy arrays to PyTorch tensors
        xs = torch.from_numpy(self.x[i:i + n])
        ys = torch.from_numpy(self.y[i:i + n])
        self.current += n
        return xs, ys
