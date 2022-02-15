import numpy as np
import os
from os.path import split
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List
import utils.constants as consts
from utils.helpers import get_paths_to_wav_files


def create_data_frame_from_path(path_to_dataset: str) -> DataFrame:
    """
    :param: path_to_audio: Path to root folder of a dataset.
    :return:

    Creates a data frame from a path to an audio dataset.
    Example of naming convention audio files must adhere too: "reed_C#_004805_segment_0.wav"

    The column created are:
    index | path_to_data | instrument_label (one-hot-encoded) | pitch_label (one-hot-encoded)
    """

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    # One hot encoded labels.
    instrument_labels: np.ndarray = get_instrument_encodings(wav_paths, instrument_classes)
    pitch_labels: np.ndarray = get_pitch_encodings(wav_paths)

    df = DataFrame.from_dict(
        {
            'path': wav_paths,
            'instrument_label': [label for label in instrument_labels],
            'pitch_label': [label for label in pitch_labels]
        }
    )

    return df


def get_pitch_encodings(wav_paths: List[str]) -> np.ndarray:
    """
    :param: wav_paths: Paths to wav_files. Must follow this naming convention: "reed_C#_004805_segment_0.wav"
    :return:

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

    return one_hot_encoded_labels


def get_instrument_encodings(wav_paths: List[str], classes: List[str]) -> np.ndarray:
    """
    :param: path_to_audio: Path to top level of dataset.
    :param: wav_paths: Path to each .wav file.
    :return: Number of classes as well as the labels

    Create labels for each wav file corresponding to its instrument.
    """

    # Encode labels.
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    labels: np.ndarray = label_encoder.transform([split(x)[0].split('/')[-1] for x in wav_paths])
    one_hot_encoded_labels: np.ndarray = to_categorical(labels, num_classes=len(classes))

    return one_hot_encoded_labels
