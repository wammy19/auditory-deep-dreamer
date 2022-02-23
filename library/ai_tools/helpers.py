import os
from os.path import split
from random import shuffle
from typing import List, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import utils.constants as consts
from utils.helpers import get_paths_to_wav_files


def create_data_frame_from_path(path_to_dataset: str, num_of_each_class: int = 50) -> DataFrame:
    """
    :param: path_to_audio: Path to root folder of a dataset.
    :return:

    Creates a data frame from a path to an audio dataset.
    Example of naming convention audio files must adhere too: "reed_C#_004805_segment_0.wav"

    The column created are:
    index | path_to_data | instrument_label (one-hot-encoded) | pitch_label (one-hot-encoded)
    """

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset, num_of_each_class)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    shuffle(wav_paths)

    # One hot encoded labels.
    encoded_pitch_labels, pitch_labels = get_pitch_encodings(wav_paths)  # type: np.ndarray, List[str]
    encoded_instrument_labels, instrument_labels = get_instrument_encodings(
        wav_paths, instrument_classes)  # type: np.ndarray, List[str]

    df = DataFrame.from_dict(
        {
            'path': wav_paths,
            'instrument': [label for label in instrument_labels],
            'pitch': [label for label in pitch_labels],
            'instrument_label': [label for label in encoded_instrument_labels],
            'pitch_label': [label for label in encoded_pitch_labels],
        }
    )

    return df


def get_pitch_encodings(wav_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    :param: wav_paths: Paths to wav_files. Must follow this naming convention: "reed_C#_004805_segment_0.wav"
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


def get_instrument_encodings(wav_paths: List[str], classes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    :param: path_to_audio: Path to top level of dataset.
    :param: wav_paths: Path to each .wav file.
    :return: Returns one hot encoded instrument labels, as well as the decoded instrument labels as strings.

    Create labels for each wav file corresponding to its instrument.
    """

    # Encode labels.
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    pre_encoded_labels: List[str] = [split(x)[0].split('/')[-1] for x in wav_paths]
    encoded_labels: np.ndarray = label_encoder.transform(pre_encoded_labels)
    one_hot_encoded_labels: np.ndarray = to_categorical(encoded_labels, num_classes=len(classes))

    return one_hot_encoded_labels, pre_encoded_labels


def decode_pitch(label: np.ndarray) -> str:
    """
    :param: label: One hot encoded pitch label. Shape must be (12,)
    :return:

    Decodes a one hot encoded vector representing pitch into a string.
    Example: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> "A"
    """

    pitch_index: int = int(np.argmax(label))
    pitch: str = consts.SORTED_NOTE_TABLE[pitch_index]

    return pitch
