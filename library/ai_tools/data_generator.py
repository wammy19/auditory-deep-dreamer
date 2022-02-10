from __future__ import annotations
from glob import glob
import numpy as np
from librosa import load
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, Sequence
from typing import List, Optional, Tuple, Union
import utils.constants as consts


class DataGenerator(Sequence):
    """
    Learning resources:
    DataGenerator inspired from: https://www.youtube.com/watch?v=OUHU7K_dD30
    """


    def __init__(
            self,
            wav_paths: List[str],
            instrument_labels: Optional[np.ndarray],
            n_instrument_classes: Optional[int],
            pitch_labels: Optional[np.ndarray],
            n_pitch_classes: Optional[int],
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
            include_instrument_label: bool = True,
            include_pitch_labels: bool = False

    ):
        # Data.
        self._instrument_labels = instrument_labels
        self._n_instrument_classes = n_instrument_classes
        self._pitch_labels = pitch_labels
        self._n_pitch_classes = n_pitch_classes

        # Settings.
        self._sample_rate = sample_rate
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._indexes: Optional[np.array] = None  # Gets defined in '.on_epoch_end()'

        # Labels.
        self._include_instrument_label = include_instrument_label
        self._include_pitch_labels = include_pitch_labels

        # Misc.
        self._wav_paths = wav_paths
        self.on_epoch_end()


    def __len__(self) -> int:
        """
        :return: int
        """

        return int(np.floor(len(self._wav_paths) / self._batch_size))


    def _get_instrument_encoded_labels(
            self,
            indexes: List[int]
    ) -> Tuple[Union[List[int], None], Union[np.array, None]]:
        """
        :param indexes:
        :return:
        """

        instrument_labels: Optional[List[int]] = None
        instrument_y: Optional[np.array] = None

        if self._include_instrument_label:
            instrument_labels = [self._instrument_labels[i] for i in indexes]
            instrument_y: np.array = np.empty((self._batch_size, self._n_instrument_classes), dtype=np.float32)

        return instrument_labels, instrument_y


    def _get_pitch_encoded_labels(self, indexes: List[int]) -> Tuple[Union[List[int], None], Union[np.array, None]]:
        """
        :param indexes:
        :return:
        """

        pitch_labels: Optional[List[int]] = None
        pitch_y: Optional[np.array] = None

        if self._include_pitch_labels:
            pitch_labels = [self._pitch_labels[i] for i in indexes]
            pitch_y = np.empty((self._batch_size, self._n_pitch_classes), dtype=np.float32)

        return pitch_labels, pitch_y


    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.array], Tuple[np.ndarray, np.array, np.ndarray]]:
        """
        :param: index: int
        :return:
        """

        indexes: List[int] = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]
        wav_paths: List[str] = [self._wav_paths[i] for i in indexes]

        # Get encoded labels.
        # Warning, these could return "None".
        instrument_labels, instrument_y = self._get_instrument_encoded_labels(indexes)
        pitch_labels, pitch_y = self._get_instrument_encoded_labels(indexes)

        X: np.ndarray = np.empty((self._batch_size, self._sample_rate, 1), dtype=np.float32)

        # Returns both instrument and pitch labels.
        if self._include_pitch_labels and self._include_pitch_labels:
            for i, (path, instrument_label, pitch_label) in enumerate(zip(wav_paths, instrument_labels, pitch_labels)):
                X[i,] = load(path, mono=True)[0].reshape(-1, 1)
                instrument_y[i,] = to_categorical(instrument_label, num_classes=self._n_instrument_classes)
                pitch_y[i,] = to_categorical(pitch_label, num_classes=self._n_pitch_classes)

            return X, instrument_y, pitch_y

        # Returns only pitch labels.
        elif self._include_pitch_labels:
            for i, (path, pitch_label) in enumerate(zip(wav_paths, pitch_labels)):
                X[i,] = load(path, mono=True)[0].reshape(-1, 1)
                pitch_y[i,] = to_categorical(pitch_label, num_classes=self._n_pitch_classes)

            return X, pitch_y

        # Returns only instrument labels.
        elif self._include_instrument_label:
            for i, (path, instrument_labels) in enumerate(zip(wav_paths, instrument_labels)):
                X[i,] = load(path, mono=True)[0].reshape(-1, 1)
                instrument_y[i,] = to_categorical(instrument_labels, num_classes=self._n_pitch_classes)

            return X, instrument_y


    def on_epoch_end(self) -> None:
        """
        :return: None

        Shuffles wave paths, so they get loaded in different batches for each epoch.
        """

        self._indexes = np.arange(len(self._wav_paths))

        if self._shuffle:
            np.random.shuffle(self._indexes)


    @staticmethod
    def _get_pitch_labels(wav_paths) -> Tuple[int, np.ndarray]:
        """
        :param: wav_paths: Paths to wav_files. Must follow this name convention: "reed_C#_004805_segment_0.wav"
        :return:

        Create labels for each sample's pitch.
        """

        # Pitch encoding.
        pitch_classes: List[str] = sorted(consts.NOTE_TABLE)  # Example: ['C', 'A#']
        n_pitch_classes: int = len(consts.NOTE_TABLE)
        pitch_label_encoder = LabelEncoder()
        pitch_label_encoder.fit(pitch_classes)

        pre_encode_pitch_labels: List[str] = []

        for _path in wav_paths:
            sample: str = os.path.split(_path)[1]

            if 'phil' in sample:
                pre_encode_pitch_labels.append(sample.split('_')[2])

            else:
                pre_encode_pitch_labels.append(sample.split('_')[1])

        pitch_labels: np.ndarray = pitch_label_encoder.transform(pre_encode_pitch_labels)

        return n_pitch_classes, pitch_labels


    @staticmethod
    def _get_instrument_labels(path_to_audio: str, wav_paths: List[str]) -> Tuple[int, np.ndarray]:
        """
        :param: path_to_audio: Path to top level of dataset.
        :param: wav_paths: Path to each .wav file.
        :return: Number of classes as well as the labels

        Create labels for each wav file corresponding to its instrument.
        """

        classes: List[str] = sorted(os.listdir(path_to_audio))  # Example: ['string', 'reed']
        n_classes: int = len(classes)

        # Encode labels.
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        labels: np.ndarray = label_encoder.transform([os.path.split(x)[0].split('/')[-1] for x in wav_paths])

        return n_classes, labels


    @classmethod
    def from_path_to_audio(
            cls,
            path_to_audio: str,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
            include_instrument_label: bool = True,
            include_pitch_labels: bool = False
    ) -> DataGenerator:
        """
        :param: path_to_audio: str - Path to root folder of audio files.
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

        # Gather paths to each '.wav' file.
        wav_paths: List[str] = glob(f'{path_to_audio}/**', recursive=True)
        wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

        # Pitch labels.
        pitch_labels: Optional[np.ndarray] = None
        n_pitch_classes: Optional[int] = None

        if include_pitch_labels:
            n_pitch_classes, pitch_labels = cls._get_pitch_labels(wav_paths)  # type: int, np.ndarray

        # Instrument labels.
        instrument_labels: Optional[np.ndarray] = None
        n_instrument_classes: Optional[int] = None

        # Include instrument labels by default.
        if include_instrument_label or include_pitch_labels is False:
            n_instrument_classes, instrument_labels = cls._get_instrument_labels(path_to_audio,
                                                                                 wav_paths)  # type: int, np.ndarray

        return cls(
            wav_paths,
            instrument_labels,
            n_instrument_classes,
            pitch_labels,
            n_pitch_classes,
            batch_size,
            sample_rate,
            shuffle,
            include_instrument_label,
            include_pitch_labels
        )
