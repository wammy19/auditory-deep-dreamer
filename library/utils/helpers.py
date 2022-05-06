import os
import re
from dataclasses import dataclass
from os.path import join
from random import shuffle
from typing import List

import numpy as np
import yaml
from librosa import load
from librosa.util import fix_length

import utils.constants as consts

# Patterns.
unix_url_substring_pattern: re.Pattern = re.compile(r'([^/]+)(?=/[^/]+/?$)')  # Example: /home/ada/SELECTION/dev/
unix_url_end_filename_pattern: re.Pattern = re.compile(r'^.+?(?=\.)')  # Example: /home/ada/audio-files/SELECTION
note_pattern: re.Pattern = re.compile(r'[A-G*#]')


@dataclass
class Data:
    """
    Little data structure for holding data and labels.
    """

    raw_audio: np.ndarray
    label: int


def load_data(path_to_audio: str) -> List[Data]:
    """
    :param path_to_audio: Path to a directory full of .wav files.
    :return
    """

    samples: List[Data] = []  # Little data structure for holding raw audio data and it's label.

    for label, _class in enumerate(os.listdir(path_to_audio)):
        for file in os.listdir(f'{path_to_audio}/{_class}'):
            sample: np.ndarray = load(
                f'{path_to_audio}/{_class}/{file}',
                duration=1.0,
                mono=True,
                sr=consts.SAMPLE_RATE
            )[0]

            sample = fix_length(sample, size=consts.SAMPLE_RATE)  # Pad audio with 0's if it's less than a second.
            samples.append(Data(sample, label))

    return samples


def midi_number_to_note(number: int) -> str:
    """
    :param number: Midi note number in the range of 0-127.
    :return str

    A table of notes that can be queried using a midi number.
    Source: https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
    """

    note_in_octave = len(consts.NOTE_TABLE)

    if number > 127 or number < 0:
        raise ValueError("Number must be in the range of 0-127")

    note: str = consts.NOTE_TABLE[number % note_in_octave]
    return note


def get_paths_to_wav_files(path_to_dataset: str, num_of_each_class: int = 50) -> List[str]:
    """
    :param path_to_dataset: Path to a directory containing wav files. Paths to wav files are collected recursively.
    :param num_of_each_class: Number of paths wanted for each instrument. If the number is greater than the number of paths there are, all the paths will be used.
    :return: List of absolute paths to wav files.
    """

    ontology: List[str] = sorted(os.listdir(path_to_dataset))
    wav_paths: List[str] = []

    for instrument in ontology:
        num_of_paths_to_get: int = num_of_each_class
        path_to_instrument: str = join(path_to_dataset, instrument)
        paths: List[str] = os.listdir(path_to_instrument)

        shuffle(paths)

        # Use all the paths there are if the number given was greater than the number of paths there are.
        if num_of_paths_to_get > len(paths):
            num_of_paths_to_get = len(paths)

        for i in range(num_of_paths_to_get):
            wav_paths.append(join(path_to_instrument, paths[i]))

    return wav_paths


def read_yaml_config(path_to_yaml: str = '/home/andrea/dev/uni/auditory-deepdream/config.yml') -> dict:
    """
    :param path_to_yaml: Absolute path to a yml file.
    :return dict with all the key pairs contained in the yaml file that is loaded.
    """

    with open(path_to_yaml, 'r') as file_handler:
        config: dict = yaml.safe_load(file_handler)

    return config
