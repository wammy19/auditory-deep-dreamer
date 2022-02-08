from __future__ import annotations
from glob import glob
import numpy as np
from librosa import load
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, Sequence
from typing import List, Optional, Tuple
import utils.constants as consts


class DataGenerator(Sequence):
    """
    Learning resources:
    DataGenerator inspired from: https://www.youtube.com/watch?v=OUHU7K_dD30
    """


    def __init__(
            self,
            wav_paths: List[str],
            labels: np.ndarray,
            n_classes: int,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True
    ):
        self._wav_paths = wav_paths
        self._labels = labels
        self._sample_rate = sample_rate
        self._n_classes = n_classes
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._indexes: Optional[np.array] = None  # Gets defined in '.on_epoch_end()'
        self.on_epoch_end()


    def __len__(self) -> int:
        """
        :return: int
        """

        return int(np.floor(len(self._wav_paths) / self._batch_size))


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.array]:
        """
        :param: index: int
        :return: Tuple[np.ndarray, np.array] - (data, label)

        Loads a batch of audio data with it's corresponding label and returns it.
        """

        indexes: List[int] = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]
        wav_paths: List[str] = [self._wav_paths[i] for i in indexes]
        labels: List[int] = [self._labels[i] for i in indexes]

        # generate a batch of time data.
        X: np.ndarray = np.empty((self._batch_size, self._sample_rate, 1), dtype=np.float32)
        y: np.array = np.empty((self._batch_size, self._n_classes), dtype=np.float32)

        # Load audio and one-hot encoding of labels.
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            X[i,] = load(path, mono=True)[0].reshape(-1, 1)
            y[i,] = to_categorical(label, num_classes=self._n_classes)

        return X, y


    def on_epoch_end(self) -> None:
        """
        :return: None

        Shuffles wave paths, so they get loaded in different batches for each epoch.
        """

        self._indexes = np.arange(len(self._wav_paths))

        if self._shuffle:
            np.random.shuffle(self._indexes)


    @staticmethod
    def _get_pitch_labels() -> np.ndarray:
        pass


    @classmethod
    def from_path_to_audio(
            cls,
            path_to_audio: str,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True
    ) -> DataGenerator:
        """
        :param: path_to_audio: str - Path to root folder of audio files.
        :param: batch_size: int - Number of '.wav' files to load in a batch.
        :param: sample_rate: Sample rate of audio files, must be the same for all files.
        :param: shuffle: Randomly shuffle data after each epoch.
        :return: Tuple[DataGenerator, DataGenerator] - (train_generator, validation_generator)

        Returns an audio DataGenerator class from a given path to a root folder that contains the ontology with audio
        files.

        Example of file structure:

        root
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
        n_classes: int = len(classes)

        # Encode labels.
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        labels: np.ndarray = label_encoder.transform([os.path.split(x)[0].split('/')[-1] for x in wav_paths])

        # TODO: Add pitch labels.
        cls._get_pitch_labels()

        return cls(
            wav_paths,
            labels,
            n_classes,
            batch_size,
            sample_rate,
            shuffle
        )
