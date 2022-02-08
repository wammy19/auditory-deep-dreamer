"""
Concurrently pre-processes audio for training.
"""

from concurrent.futures import ThreadPoolExecutor
from librosa import load
from librosa.effects import trim
import numpy as np
import os
from os.path import join
import re
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/index.html#soundfile.write
from typing import List, Generator
from tqdm import tqdm
from utils.audio_processors import segment_signal
import utils.constants as consts


def process_audio(path_to_wavs: str, path_for_saving_processed_audio: str):
    """
    :param: path_to_wavs:
    :param: path_for_saving_processed_audio:
    :return:

    Processes the audio and saves it in a new directory.
    """

    inst_class: str = re.findall(r'([^/]+)/?$', path_to_wavs)[0]
    writing_path: str = join(path_for_saving_processed_audio, inst_class)
    writing_path_np: str = join(f'{path_for_saving_processed_audio}_numpy', inst_class)  # For saving np arrays.

    os.makedirs(writing_path, exist_ok=True)
    os.makedirs(writing_path_np, exist_ok=True)

    for file in tqdm(os.listdir(path_to_wavs), desc=f'{inst_class}'):  # wav files.
        path_to_sample: str = join(path_to_wavs, file)

        # Load in audio and process it.
        sample: np.ndarray = load(path_to_sample, consts.SAMPLE_RATE, mono=True)[0]
        sample = trim(sample, top_db=consts.TRIM_DB)[0]
        segmented_signal: List[np.ndarray] = segment_signal(sample)

        # Save audio.
        for i, audio in enumerate(segmented_signal):
            file_for_writing_name: str = f'{os.path.splitext(file)[0]}_segment_{i}.wav'
            file_for_writing_name_np: str = f'{os.path.splitext(file)[0]}_segment_{i}'

            # Save both audio and serialized numpy array.
            sf.write(join(writing_path, file_for_writing_name), audio, consts.SAMPLE_RATE)
            np.save(join(writing_path_np, file_for_writing_name_np), audio.reshape(-1, 1))


def generate_paths_to_wav_files(_path_to_dataset: str) -> Generator[str, None, None]:
    """
    :param: path_to_data_set
    :return: Path to dataset.

    Generates paths to all wav files in an ontology.
    """

    for category in os.listdir(_path_to_dataset):
        path_to_classes: str = os.path.join(_path_to_dataset, category)

        yield path_to_classes


def main():
    # Settings.
    number_of_threads: int = 16
    path_to_dataset: str = '../../data-sets/nsynth'
    path_for_writing: str = '../../data-sets/processed_dataset'  # Will be created if it doesn't exist.

    paths_to_wavs: Generator[str, None, None] = generate_paths_to_wav_files(path_to_dataset)
    os.makedirs(path_for_writing, exist_ok=True)

    # Concurrently process all audio.
    with ThreadPoolExecutor(max_workers=number_of_threads) as thread_executor:
        for path in paths_to_wavs:
            thread_executor.submit(process_audio, path, path_for_writing)


if __name__ == '__main__':
    main()
