"""
Concurrently pre-processes audio for training.
"""

from concurrent.futures import ThreadPoolExecutor
from librosa import load
from librosa.effects import trim
import numpy as np
import os
import re
import soundfile as sf
from typing import List, Generator
from tqdm import tqdm  # Progress bars.
from utils.audio_processors import segment_signal
import utils.constants as consts

number_of_threads: int = 16
path_to_dataset: str = '../data-sets/philharmonia_dataset'
path_for_writing: str = '../data-sets/processed_philharmonia_dataset'


def process_audio(path_to_wavs: str, path_for_saving_processed_audio: str):
    """
    :param path_to_wavs:
    :param path_for_saving_processed_audio:
    :return:

    Processes the audio and saves it in a new directory.
    """

    data_type: str = re.findall(r'([^/]+)(?=/[^/]+/?$)', path_to_wavs)[0]  # train, test, validation
    inst_class: str = re.findall(r'([^/]+)/?$', path_to_wavs)[0]
    writing_path: str = os.path.join(path_for_saving_processed_audio, data_type, inst_class)

    os.makedirs(writing_path, exist_ok=True)

    for file in tqdm(os.listdir(path_to_wavs), desc=f'{data_type} - {inst_class}'):  # wav files.

        # Ignore other files which aren't audio files.
        if '.wav' not in file:
            pass

        path_to_sample: str = os.path.join(path_to_wavs, file)

        # Load in audio and process it.
        sample: np.ndarray = load(path_to_sample, consts.SAMPLE_RATE, mono=True)[0]
        sample = trim(sample, top_db=consts.TRIM_DB)[0]
        segmented_signal: List[np.ndarray] = segment_signal(sample)
        segment_num: int = 0

        # Save audio.
        for audio in segmented_signal:
            file_for_writing: str = f'{os.path.splitext(file)[0]}_segment_{segment_num}.wav'
            segment_num += 1

            sf.write(os.path.join(writing_path, file_for_writing), audio, consts.SAMPLE_RATE)


def generate_paths_to_wav_files(path_to_data_set: str) -> Generator[str, None, None]:
    """
    :return: Path to dataset.

    Generates the paths to wav files which for a data set that is organized like so:

    data-set
    |
    |_____validation
    |     |_________example.wav ...
    |
    |_____test
    |     |_________example.wav ...
    |
    |_____train
          |_________example.wav ...
    """

    all_paths: List[str] = []

    for folder in os.listdir(path_to_data_set):  # test, validation, and train folders.
        sub_folder_path: str = os.path.join(path_to_data_set, folder)

        for category in os.listdir(sub_folder_path):  # Instrument classes.
            path_to_classes: str = os.path.join(sub_folder_path, category)
            all_paths.append(path_to_classes)

            yield path_to_classes


paths_to_wavs: Generator[str, None, None] = generate_paths_to_wav_files(path_to_dataset)
os.makedirs(path_for_writing, exist_ok=True)

# Concurrently process all audio.
with ThreadPoolExecutor(max_workers=number_of_threads) as thread_executor:
    for path in paths_to_wavs:
        thread_executor.submit(process_audio, path, path_for_writing)
