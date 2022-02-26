import os
from typing import List

import numpy as np
from pandas import DataFrame

from ai_tools import DataGenerator
from ai_tools.helpers import create_data_frame_from_path, decode_instrument, decode_pitch, get_instrument_encodings, \
    get_pitch_encodings
from utils.constants import SORTED_NOTE_TABLE
from utils.helpers import get_paths_to_wav_files, note_pattern, read_yaml_config, unix_url_substring_pattern

yaml_config: dict = read_yaml_config()
PATH_TO_DATASET: str = yaml_config['path_to_dataset']


def test_encoders():
    """
    :return:

    Tests the pitch and instrument one-hot-encoders match the correct file.
    """

    wav_paths: List[str] = get_paths_to_wav_files(PATH_TO_DATASET)
    instrument_classes: List[str] = sorted(os.listdir(PATH_TO_DATASET))  # Example: ['string', 'reed']

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

    Tests the data frame creator function.
    """

    number_of_each_class: int = 50
    ontology: List[str] = os.listdir(PATH_TO_DATASET)
    ontology_len: int = len(ontology)
    df: DataFrame = create_data_frame_from_path(PATH_TO_DATASET, number_of_samples_for_each_class=number_of_each_class)

    assert df['instrument'].count() == number_of_each_class * ontology_len

    for _class in ontology:
        assert len(df[df['instrument'] == 'bass'].index) == number_of_each_class


def test_data_generator():
    """
    :return:
    """

    number_of_samples_for_each_class: int = 50
    batch_size: int = 10

    # False shuffle so that we can appropriately test that the path matches the correct audio.
    data_generator = DataGenerator.from_path_to_audio(
        PATH_TO_DATASET,
        number_of_samples_for_each_class=number_of_samples_for_each_class,
        shuffle=False,
        batch_size=batch_size,
        include_pitch_labels=True
    )

    df: DataFrame = data_generator.get_data_frame  # Copy DataFrame that is stored inside the

    for batch_index in range(number_of_samples_for_each_class // batch_size):
        for i in range(batch_size):
            # Get all data from DataGenerator.
            decoded_instrument: str = decode_instrument(data_generator[batch_index][1][i])
            decoded_pitch: str = decode_pitch(data_generator[batch_index][2][i])

            df_index: int = (batch_index * 10) + i

            assert df['instrument'][df_index] == decoded_instrument
            assert df['pitch'][df_index] == decoded_pitch
