import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from os.path import join
from typing import List

import numpy as np
from tqdm import tqdm

import utils.constants as consts
from utils.audio_tools import load_and_convert_audio_into_mel_spectrogram

PATH_TO_DATASET: str = '../../datasets/processed_dataset'
PATH_TO_SERIALIZED_DATASET: str = '../../datasets/serialized_dataset'

def serialize_and_save(path_to_samples: str, path_for_saving: str, sample: str) -> None:
    """
    :param path_to_samples:
    :param path_for_saving:
    :param sample:
    :return:

    Loads audio as a numpy array and saves the numpy array.
    """

    path_to_sample: str = join(path_to_samples, sample)
    mel_spectrogram: np.ndarray = load_and_convert_audio_into_mel_spectrogram(path_to_sample)
    new_sample_name: str = sample.replace('.wav', '.npy')

    np.save(join(path_for_saving, new_sample_name), mel_spectrogram.reshape(consts.X_SHAPE))


def main():
    ontology: List[str] = os.listdir(PATH_TO_DATASET)

    for instrument in ontology:
        path_to_samples: str = join(PATH_TO_DATASET, instrument)
        path_for_saving: str = join(PATH_TO_SERIALIZED_DATASET, instrument)
        samples: List[str] = os.listdir(path_to_samples)
        num_samples: int = len(samples)

        os.makedirs(join(PATH_TO_SERIALIZED_DATASET, instrument), exist_ok=True)

        # Progress bar.
        pbar = tqdm(desc=instrument, total=num_samples)
        futures: List[Future[None]] = []

        with ThreadPoolExecutor(max_workers=16) as thread_pool_executor:
            for sample in samples:
                futures.append(
                    thread_pool_executor.submit(
                        serialize_and_save,
                        path_to_samples,
                        path_for_saving,
                        sample
                    )
                )

            for _ in as_completed(futures):
                pbar.update(1)


if __name__ == '__main__':
    main()
