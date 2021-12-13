import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import json
import numpy as np
from typing import List, Tuple


class DataGenerator(Sequential):
    """
    """


    def __init__(
            self,
            list_IDs: List,
            labels: List[int],
            batch_size: int = 32,
            dim: Tuple = (32, 32, 32),
            n_channels: int = 1,
            n_classes: int = 10,
            shuffle: bool = True
    ):
        """
        :param list_IDs:
        :param labels:
        :param batch_size:
        :param dim:
        :param n_channels:
        :param n_classes:
        :param shuffle:
        """

        super().__init__()

        # Data.
        self.list_IDs = list_IDs
        self.labels = labels

        # Parameters
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes: np.ndarray = np.array([])
        self.on_epoch_end()


    def __len__(self):
        """
        :param self:
        :return:
        """

        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """
        :param index:
        :return:
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        """
        :return:
        """

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        """
        :param list_IDs_temp:
        :return:
        """

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, to_categorical(y, num_classes=self.n_classes)
