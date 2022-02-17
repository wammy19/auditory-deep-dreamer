"""
Concurrently pre-processes audio for training.
"""

from concurrent.futures import ProcessPoolExecutor
from librosa import load
from librosa.effects import trim
import numpy as np
import os
from os import listdir
from os.path import join
import re
from sklearn.cluster import KMeans
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/index.html#soundfile.write
from typing import List
from utils.audio_processors import segment_signal
import utils.constants as consts


# Settings.
number_of_threads: int = 16
path_to_dataset: str = '/home/andrea/dev/uni/datasets/nsynth'
path_for_writing: str = '/home/andrea/dev/uni/datasets/processed_dataset'  # Will be created if it doesn't exist.
num_clusters: int = 4


def process_audio(path: str, file: str, inst_class: str, window_leap_fraction: int):
    """
    :param: path:
    :param: file:
    :return:

    Processes the audio and saves it in a new directory.
    """

    path_to_sample: str = join(path, file)

    # Load in audio and process it.
    sample: np.ndarray = load(path_to_sample, sr=consts.SAMPLE_RATE, mono=True)[0]
    sample = trim(sample, top_db=consts.TRIM_DB)[0]
    segmented_signal: List[np.ndarray] = segment_signal(sample, window_leap_fraction=window_leap_fraction)

    # Save audio.
    for i, audio in enumerate(segmented_signal):
        file_name: str = f'{os.path.splitext(file)[0]}_segment_{i}.wav'

        sf.write(join(path_for_writing, inst_class, file_name), audio, consts.SAMPLE_RATE)


def main():
    samples_path: List[dict] = []

    for instrument in listdir(path_to_dataset):
        path: str = join(path_to_dataset, instrument)

        samples_path.append({
            'path': path,
            'num_samples': len(listdir(path))
        })

    samples_path = sorted(samples_path, key=lambda x: x['num_samples'])

    # Cluster instruments based on how many samples there are.
    X: np.ndarray = np.array([[samples['num_samples']] for samples in samples_path])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

    os.makedirs(path_for_writing, exist_ok=True)

    current_cluster: int = kmeans.labels_[0]
    window_leap: int = num_clusters

    for i, sample in enumerate(samples_path):
        inst_class: str = re.findall(r'([^/]+)/?$', sample['path'])[0]
        writing_path: str = join(path_for_writing, inst_class)

        os.makedirs(writing_path, exist_ok=True)

        if kmeans.labels_[i] != current_cluster:
            window_leap -= 1
            current_cluster = kmeans.labels_[i]

        with ProcessPoolExecutor(max_workers=number_of_threads) as process_pool_executor:
            for file in os.listdir(sample['path']):
                process_pool_executor.submit(process_audio, sample['path'], file, inst_class, window_leap)


if __name__ == '__main__':
    main()
