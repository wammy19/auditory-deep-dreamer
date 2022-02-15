from ai_tools.helpers import get_instrument_encodings, get_pitch_encodings
import numpy as np
import os
from typing import List
from utils.constants import SORTED_NOTE_TABLE
from utils.helpers import unix_url_substring_pattern, note_pattern, get_paths_to_wav_files, read_yaml_config

yaml_config: dict = read_yaml_config()


def test_encoders():
    """
    :return:

    Tests the pitch and instrument one-hot-encoders match the correct file.
    """

    path_to_dataset: str = yaml_config['path_to_dataset']

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    # One hot encoded labels.
    instrument_labels: np.ndarray = get_instrument_encodings(wav_paths, instrument_classes)
    pitch_labels: np.ndarray = get_pitch_encodings(wav_paths)

    for path, instrument_label, pitch_label in zip(wav_paths, instrument_labels, pitch_labels):
        instrument: str = unix_url_substring_pattern.findall(path)[0]  # Get data from path.

        # Matches note and sharp symbol separately, so they are joined.
        note_matches: List[str] = note_pattern.findall(path)
        note: str = ''.join(note_matches)

        assert instrument_classes[np.argmax(instrument_label)] == instrument
        assert SORTED_NOTE_TABLE[np.argmax(pitch_label)] == note

        assert instrument_label.shape == (len(instrument_classes),)
        assert pitch_label.shape == (len(SORTED_NOTE_TABLE),)
