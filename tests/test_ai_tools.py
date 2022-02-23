import os
from typing import List

import numpy as np

from ai_tools.helpers import decode_pitch, get_instrument_encodings, get_pitch_encodings, create_data_frame_from_path
from utils.constants import SORTED_NOTE_TABLE
from utils.helpers import get_paths_to_wav_files, note_pattern, read_yaml_config, unix_url_substring_pattern

from pandas import DataFrame

yaml_config: dict = read_yaml_config()
path_to_dataset: str = yaml_config['path_to_dataset']


def test_encoders():
    """
    :return:

    Tests the pitch and instrument one-hot-encoders match the correct file.
    """

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    # One hot encoded labels.
    instrument_labels = get_instrument_encodings(wav_paths, instrument_classes)[0]
    pitch_labels = get_pitch_encodings(wav_paths)[0]

    for path, instrument_label, pitch_label in zip(wav_paths, instrument_labels, pitch_labels):
        instrument: str = unix_url_substring_pattern.findall(path)[0]  # Get data from path.

        # Matches note and sharp symbol separately, so they are joined.
        note_matches: List[str] = note_pattern.findall(path)
        note: str = ''.join(note_matches)

        assert instrument_classes[np.argmax(instrument_label)] == instrument
        assert decode_pitch(pitch_label) == note

        assert instrument_label.shape == (len(instrument_classes),)
        assert pitch_label.shape == (len(SORTED_NOTE_TABLE),)


def test_data_frame_creator():
    """
    :return:
    """
    number_of_each_class: int = 50
    ontology: List[str] = os.listdir(path_to_dataset)
    ontology_len: int = len(ontology)
    df: DataFrame = create_data_frame_from_path(path_to_dataset, num_of_each_class=number_of_each_class)

    assert df['instrument'].count() == number_of_each_class * ontology_len

    for _class in ontology:
        assert len(df[df['instrument'] == 'bass'].index) == number_of_each_class
