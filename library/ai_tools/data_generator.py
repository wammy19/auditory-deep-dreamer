from __future__ import annotations
from glob import glob
import numpy as np
import os
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from typing import List, Optional, Tuple
import utils.constants as consts


class DataGenerator(Sequence):

    def __init__(
            self,
            wav_paths: List[str],
            labels: np.ndarray,
            n_classes: int,
            batch_size=16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True
    ):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes: Optional[np.array] = None  # Get's defined in '.on_epoch_end()'
        self.on_epoch_end()


    def __len__(self) -> int:
        """
        :return: int
        """

        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.array]:
        """
        :param index: int
        :return: Tuple[np.ndarray, np.array] - (data, label)

        Loads a batch of audio data with it's corresponding label and returns it.
        """

        indexes: List[int] = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        wav_paths: List[str] = [self.wav_paths[i] for i in indexes]
        labels: List[int] = [self.labels[i] for i in indexes]

        # generate a batch of time data
        X: np.ndarray = np.empty((self.batch_size, int(self.sample_rate), 1), dtype=np.float32)
        y: np.array = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            wav: np.ndarray = wavfile.read(path)[1]
            X[i,] = wav.reshape(-1, 1)
            y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, y


    def on_epoch_end(self) -> None:
        """
        :return: None

        Shuffles wave paths so they get loaded in different batches for each epoch.
        """

        self.indexes = np.arange(len(self.wav_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)


    @classmethod
    def from_path_to_audio(
            cls,
            path_to_audio: str,
            validation_split: float = 0.1,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True
    ) -> Tuple[DataGenerator, DataGenerator]:
        """
        :param path_to_audio: str - Path to root folder of audio files.
        :param validation_split: float - Number between 0 - 1. Default is 0.1 (10%).
        :param batch_size: int - Number of '.wav' files to load in a batch.
        :param sample_rate: Sample rate of audio files, must be the same for all files.
        :param shuffle: Randomly shuffle data after each epoch.
        :return: Tuple[DataGenerator, DataGenerator] - (train_generator, validation_generator)

        Returns two DataGenerator classes for training and validation from a given path to a root folder that contains
        ontology with audio files.

        Example of file structure:

        root-folder
        |
        |_________strings
        |         |
        |         |____string_1.wav ...
        |
        |_________reed
                  |
                  |____reed_1.wav ...
        """

        # Gather paths to each '.wav' file.
        wav_paths: List[str] = glob('{}/**'.format(path_to_audio), recursive=True)
        wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

        classes: List[str] = sorted(os.listdir(path_to_audio))  # Example: ['string', 'reed']
        n_classes: int = len(os.listdir(path_to_audio))

        # Encode labels.
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        labels: np.ndarray = label_encoder.transform([os.path.split(x)[0].split('/')[-1] for x in wav_paths])

        # Split train data for validation set.
        wav_train, wav_val, label_train, label_val = train_test_split(
            wav_paths,
            labels,
            test_size=validation_split,
            random_state=0
        )

        return \
            cls(  # Train data generator.
                wav_train,
                label_train,
                n_classes,
                batch_size,
                sample_rate,
                shuffle

            ), \
            cls(  # Validation data generator.
                wav_val,
                label_val,
                n_classes,
                batch_size,
                sample_rate,
                shuffle
            )
