from __future__ import annotations
from .helpers import create_data_frame_from_path
import numpy as np
from librosa import load
import os
from pandas import DataFrame
from tensorflow.keras.utils import Sequence
from typing import List, Optional, Union, Tuple
import utils.constants as consts


class DataGenerator(Sequence):

    def __init__(
            self,
            data_frame: DataFrame,
            batch_size: int = 16,
            num_instrument_classes: int = 10,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
            include_pitch_labels: bool = False

    ):
        # Data frame must contain these columns:
        # path | instrument_label (one-hot-encoded) | pitch_label (one-hot-encoded)
        self._df = data_frame

        # Settings.
        self._sample_rate = sample_rate
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._indexes: Optional[np.array] = None  # Gets defined in '.on_epoch_end()'

        # Labels.
        self._include_pitch_labels = include_pitch_labels
        self._num_instrument_classes = num_instrument_classes

        # Misc.
        self.on_epoch_end()


    # ----------------------------------------------- Virtual functions -----------------------------------------------


    def __len__(self) -> int:
        """
        :return: int
        """

        return int(np.floor(len(self._df.index) / self._batch_size))


    def __getitem__(
            self,
            index: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        :param: index: int
        :return:
        """

        # Gather indices.
        indexes: List[int] = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]

        wav_paths: List[str] = [self._df.loc[i]['path'] for i in indexes]
        instrument_labels: List[np.ndarray] = [self._df.loc[i]['instrument_label'] for i in indexes]
        pitch_labels: List[np.ndarray] = [self._df.loc[i]['pitch_label'] for i in indexes]

        # Initialize numpy arrays with shape.
        X: np.ndarray = np.empty((self._batch_size, self._sample_rate, 1), dtype=np.float32)
        instrument_y: np.ndarray = np.empty((self._batch_size, self._num_instrument_classes), dtype=np.float32)
        pitch_y: np.ndarray = np.empty((self._batch_size, 12), dtype=np.float32)

        # Populate arrays with data and labels.
        for i, path in enumerate(wav_paths):
            X[i, ] = load(path, mono=True)[0].reshape(-1, 1)
            instrument_y[i, ] = np.array(instrument_labels[i])
            pitch_y[i, ] = np.array(pitch_labels[i])

        if self._include_pitch_labels:
            return X, instrument_y, pitch_y

        else:
            return X, instrument_y


    def on_epoch_end(self) -> None:
        """
        :return: None

        Shuffles indices randomly at the end of an epoch.
        """

        self._indexes = np.arange(len(self._df.index))

        if self._shuffle:
            np.random.shuffle(self._indexes)


    # -----------------------------------------------------------------------------------------------------------------


    @property
    def get_data_frame(self) -> DataFrame:
        return self._df


    @classmethod
    def from_path_to_audio(
            cls,
            path_to_audio: str,
            include_pitch_labels: bool = False,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
    ) -> DataGenerator:
        """
        :param: path_to_audio: str - Path to root folder of audio files.
        :param: include_instrument_label: bool - Will generate instrument labels if set to True.
        :param: include_pitch_labels: bool - Will generate pitch labels if set to True.
        :param: batch_size: int - Number of '.wav' files to load in a batch.
        :param: sample_rate: Sample rate of audio files, must be the same for all files.
        :param: shuffle: Randomly shuffle data after each epoch.
        :return: DataGenerator

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

        df: DataFrame = create_data_frame_from_path(path_to_audio)
        num_instrument_classes: int = len(os.listdir(path_to_audio))

        return cls(
            df,
            batch_size,
            num_instrument_classes,
            sample_rate,
            shuffle,
            include_pitch_labels
        )
