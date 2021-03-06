import os
from os.path import split
from random import shuffle
from typing import List, Optional, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import utils.constants as consts
from utils.helpers import get_paths_to_wav_files


def create_data_frame_from_path(path_to_dataset: str, number_of_samples_for_each_class: int = 50) -> DataFrame:
    """
    :param path_to_dataset: Path to root folder of a dataset.
    :param number_of_samples_for_each_class:
    :return:

    Creates a data frame from a path to an audio dataset.
    Example of naming convention audio files must adhere too: "reed_C#_004805_segment_0.wav"

    The column created are:
    index | path_to_data | instrument_label (one-hot-encoded) | pitch_label (one-hot-encoded)
    """

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset, number_of_samples_for_each_class)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    shuffle(wav_paths)

    # One hot encoded labels.
    encoded_pitch_labels, pitch_labels = get_pitch_encodings(wav_paths)  # type: np.ndarray, List[str]
    encoded_instrument_labels, instrument_labels = get_instrument_encodings(
        wav_paths,
        instrument_classes
    )  # type: np.ndarray, List[str]

    df = DataFrame.from_dict(
        dict(
            path=wav_paths,
            instrument=[label for label in instrument_labels],
            pitch=[label for label in pitch_labels],
            instrument_label=[label for label in encoded_instrument_labels],
            pitch_label=[label for label in encoded_pitch_labels],
        )
    )

    return df


def get_pitch_encodings(wav_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    :param wav_paths: Paths to wav_files. Must follow this naming convention: "reed_C#_004805_segment_0.wav"
    :return: Returns one hot encoded pitch labels as well as the decoded pitch labels as strings.

    Create labels for each sample's pitch.
    """

    # Pitch encoding.
    pitch_classes: List[str] = consts.SORTED_NOTE_TABLE  # Example: ['C', 'A#']
    pitch_label_encoder = LabelEncoder()
    pitch_label_encoder.fit(pitch_classes)

    pre_encode_pitch_labels: List[str] = []

    for _path in wav_paths:
        sample: str = split(_path)[1]

        # Philharmonia orchestra samples are tagged with 'phil'.
        if 'phil' in sample:
            pre_encode_pitch_labels.append(sample.split('_')[2])

        else:
            pre_encode_pitch_labels.append(sample.split('_')[1])

    labels: np.ndarray = pitch_label_encoder.transform(pre_encode_pitch_labels)
    one_hot_encoded_labels: np.ndarray = to_categorical(labels, num_classes=len(consts.NOTE_TABLE))

    return one_hot_encoded_labels, pre_encode_pitch_labels


def get_instrument_encodings(wav_paths: List[str], ontology: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    :param wav_paths: Path to each .wav file.
    :param ontology: List of instruments. Example: ['reed', 'string', 'keyboards']
    :return: Returns one hot encoded instrument labels, as well as the decoded instrument labels as strings.

    Create labels for each wav file corresponding to its instrument.
    """

    # Encode labels.
    label_encoder = LabelEncoder()
    label_encoder.fit(ontology)
    pre_encoded_labels: List[str] = [split(x)[0].split('/')[-1] for x in wav_paths]
    encoded_labels: np.ndarray = label_encoder.transform(pre_encoded_labels)
    one_hot_encoded_labels: np.ndarray = to_categorical(encoded_labels, num_classes=len(ontology))

    return one_hot_encoded_labels, pre_encoded_labels


def decode_instrument(label: np.ndarray, ontology: Optional[List[str]] = None) -> str:
    """
    :param label:
    :param ontology:
    :return:
    """

    if ontology is None:
        ontology = consts.INSTRUMENT_ONTOLOGY

    assert len(ontology) == label.shape[0]

    instrument_index: int = int(np.argmax(label))
    instrument: str = ontology[instrument_index]

    return instrument


def decode_pitch(label: np.ndarray) -> str:
    """
    :param label: One hot encoded pitch label. Shape must be (12,)
    :return:

    Decodes a one hot encoded vector representing pitch into a string.
    Example: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> "A"
    """

    assert len(label) == len(consts.SORTED_NOTE_TABLE)

    pitch_index: int = int(np.argmax(label))
    pitch: str = consts.SORTED_NOTE_TABLE[pitch_index]

    return pitch


def split_stratified_into_train_val_test(
        df_input: DataFrame,
        stratify_column_name: str = 'instrument',
        frac_train: float = 0.6,
        frac_val: float = 0.2,
        frac_test: float = 0.2,
        random_state: Optional[int] = None
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    :param df_input: pandas.DataFrame for splitting.
    :param stratify_column_name: Column name for balancing data.
    :param frac_train: Percent of data wanted for training. A float ranging from 0-1
    :param frac_val: Percent of data wanted for validation. A float ranging from 0-1
    :param frac_test: Percent of data wanted for testing. A float ranging from 0-1
    :param random_state: Random shuffle for sklearn.model.train_test_split. If None this will be set randomly.
    :return: Returns 3 DataFrames for testing, validation, and training.

    This function creates 3 DataFrame split up for training, validation, and testing purposes given a
    give DataFrame as input.

    This function is not my code.
    Learning resources: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    """

    if (frac_train + frac_val + frac_test) != 1.0:
        raise ValueError(f'fractions {frac_train}, {frac_val}, {frac_test} do not add up to 1.0')

    if stratify_column_name not in df_input.columns:
        raise ValueError(f'{stratify_column_name} is not a column in the dataframe')

    X: DataFrame = df_input  # Contains all columns.
    y: DataFrame = df_input[[stratify_column_name]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X,
        y,
        stratify=y,
        test_size=(1.0 - frac_train),
        random_state=random_state
    )  # type: DataFrame, DataFrame, DataFrame, DataFrame

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)

    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state
    )  # type: DataFrame, DataFrame, DataFrame, DataFrame

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test
