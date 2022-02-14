import numpy as np
from os.path import split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List
import utils.constants as consts


def get_pitch_encodings(wav_paths: List[str]) -> np.ndarray:
    """
    :param: wav_paths: Paths to wav_files. Must follow this name convention: "reed_C#_004805_segment_0.wav"
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
