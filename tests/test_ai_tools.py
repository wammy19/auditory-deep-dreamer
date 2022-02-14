from ai_tools.encoders import get_instrument_encodings, get_pitch_encodings
from glob import glob
import numpy as np
import os
from os.path import abspath
import re
from typing import List
from utils.constants import SORTED_NOTE_TABLE


def test_encoders():
    """
    :return:

    Tests the pitch and instrument one-hot-encoders.
    """

    path_to_dataset: str = abspath('/home/andrea/dev/uni/data-sets/processed_dataset')

    # Patterns.
    unix_url_substring_pattern: re.Pattern = re.compile(r'([^/]+)(?=/[^/]+/?$)')
    note_pattern: re.Pattern = re.compile(r'[A-G*#]')

    # Gather paths to each '.wav' file.
    wav_paths: List[str] = sorted(glob(f'{path_to_dataset}/**', recursive=True))
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]

    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    # One hot encoded labels.
    instrument_labels: np.ndarray = get_instrument_encodings(wav_paths, instrument_classes)
    pitch_labels: np.ndarray = get_pitch_encodings(wav_paths)

    for path, instrument_label, pitch_label in zip(wav_paths, instrument_labels, pitch_labels):
        instrument: str = unix_url_substring_pattern.findall(path)[0]
        note_matches: List[str] = note_pattern.findall(path)
        note: str = ''.join(note_matches)

        assert instrument_classes[np.argmax(instrument_label)] == instrument
        assert SORTED_NOTE_TABLE[np.argmax(pitch_label)] == note
