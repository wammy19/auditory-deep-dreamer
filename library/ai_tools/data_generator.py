from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from os.path import join
from typing import List, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame
from tensorflow.keras.utils import Sequence

import utils.constants as consts
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test


class DataGenerator(Sequence):
    """
    Data generator class loads in raw audio and encodes the signal into a mel spectrogram in a set batch size.
    """


    # =================================================================================================================
    # ---------------------------------------------- Class Constructors -----------------------------------------------
    # =================================================================================================================

    def __init__(
            self,
            data_frame: DataFrame,
            batch_size: int = 16,
            num_instrument_classes: int = 10,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
            include_pitch_labels: bool = False,
            num_thread_workers: int = 8  # 16 Threads.
    ):
        # Data frame must contain these columns:
        # path | instrument | pitch | instrument_label (one-hot-encoded) | pitch_label (one-hot-encoded)
        self._df = data_frame.reset_index(drop=True)  # Ensure indexes are ordered to begin with.

        # Settings.
        self._sample_rate = sample_rate
        self._batch_size = batch_size
        self._shuffle = shuffle

        # Labels.
        self._indexes: Optional[np.array] = None  # Gets defined in 'self.on_epoch_end()'
        self._include_pitch_labels = include_pitch_labels
        self._num_instrument_classes = num_instrument_classes

        # Multi process pool for data loading.
        self._process_pool_executor = ProcessPoolExecutor(max_workers=num_thread_workers)

        # Function calls.
        self.on_epoch_end()


    @classmethod
    def from_path_to_audio(
            cls,
            path_to_audio: str,
            include_pitch_labels: bool = False,
            batch_size: int = 16,
            sample_rate: int = consts.SAMPLE_RATE,
            shuffle: bool = True,
            number_of_samples_for_each_class: int = 50,
            num_thread_workers: int = 8
    ) -> DataGenerator:
        """
        :param path_to_audio: str - Path to root folder of audio files.
        :param include_pitch_labels: bool - Will generate pitch labels if set to True.
        :param batch_size: int - Number of '.wav' files to load in a batch.
        :param sample_rate: Sample rate of audio files, must be the same for all files.
        :param shuffle: Randomly shuffle data after each epoch.
        :param number_of_samples_for_each_class: Number of samples wanted for each instrument.
        :param num_thread_workers: Number of multi-threading workers for loading in data.
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
                  |11
                  |____reed_1.wav ...
        """

        num_instrument_classes: int = len(os.listdir(path_to_audio))
        df: DataFrame = create_data_frame_from_path(
            path_to_audio,
            number_of_samples_for_each_class=number_of_samples_for_each_class
        )

        return cls(
            df,
            batch_size,
            num_instrument_classes,
            sample_rate,
            shuffle,
            include_pitch_labels,
            num_thread_workers
        )


    @classmethod
    def create_train_val_test_data_generators(
            cls,
            path_to_dataset: str,
            num_of_samples_per_instrument: int = 50_000,
            path_to_logs: str = './logs',
            training_batch_size: int = 32,
            test_split: float = 0.2,
    ) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
        """
        :param path_to_dataset: String path to root folder of dataset.
        :param num_of_samples_per_instrument: Number of samples for each instrument in the ontology.
        :param path_to_logs: Path to logs. Datasets are shuffled randomly, therefore they are stored in CSV format
        for the purpose of reproducing any tests.
        :param training_batch_size: Batch size for training.
        :param test_split: Percentage of data for test and validation.
        :return: Returns training, validation, testing data generators.

        Creates a train, validation, and test DataGenerator for training.
        The sets are also logged in csv format.
        """

        # Create dataset dataframe and split it into train, validation, and test.
        df: DataFrame = create_data_frame_from_path(
            path_to_dataset,
            number_of_samples_for_each_class=num_of_samples_per_instrument
        )

        # Split dataset into train, test, and validation sets.
        df_train, df_val, df_test = split_stratified_into_train_val_test(
            df,
            frac_val=test_split,
            frac_test=test_split
        )  # type: DataFrame, DataFrame, DataFrame

        datasets: List[str] = ['train_data.csv', 'val_data.csv', 'test_data.csv']

        # for set in datasets:
        #     open(join(path_to_logs, set), 'w')

        # Store the data generator data frame for recreating the data generator if needed.
        # df_train.to_csv(join(path_to_logs, 'train_data.csv'))
        # df_val.to_csv(join(path_to_logs, 'val_data.csv'))
        # df_test.to_csv(join(path_to_logs, 'test_data.csv'))

        # Create Generators.
        train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=training_batch_size)
        validation_data_generator: DataGenerator = DataGenerator(df_val, batch_size=training_batch_size)
        test_data_generator: DataGenerator = DataGenerator(df_test, batch_size=training_batch_size)

        return train_data_generator, validation_data_generator, test_data_generator


    # =================================================================================================================
    # ----------------------------------------------- Virtual functions -----------------------------------------------
    # =================================================================================================================

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
        :param index: int
        :return:
        """

        # Gather indices.
        indexes: List[int] = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]

        # Gather data from dataframe.
        wav_paths: List[str] = [self._df.loc[i]['path'] for i in indexes]
        instrument_y: np.ndarray = np.array([self._df.loc[i]['instrument_label'] for i in indexes])
        pitch_y: np.ndarray = np.array([self._df.loc[i]['pitch_label'] for i in indexes])

        # Future data.
        X: List[np.ndarray] = []
        futures: List[Future[np.ndarray]] = []

        # Concurrently load and encode data.
        for path in wav_paths:
            futures.append(self._process_pool_executor.submit(self._load_mel_spectrogram, path))

        for future in as_completed(futures):
            X.append(future.result())

        # Convert into a numpy array.
        X: np.ndarray = np.stack(X)

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


    # =================================================================================================================
    # ----------------------------------------------- Getter functions ------------------------------------------------
    # =================================================================================================================

    @property
    def get_data_frame(self) -> DataFrame:
        """
        :return: Returns stored dataframe.
        """

        return self._df


    @property
    def get_batch_size(self) -> int:
        """
        :return: Returns batch size.
        """

        return self._batch_size


    # =================================================================================================================
    # ----------------------------------------------- Private functions -----------------------------------------------
    # =================================================================================================================

    @staticmethod
    def _load_mel_spectrogram(path) -> np.ndarray:
        """
        :return:
        """

        return np.load(path)
