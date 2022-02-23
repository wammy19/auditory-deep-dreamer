import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from typing import List

from utils.audio_tools import time_stretch_signal

# Settings.
path_to_data_to_augment: str = '/home/andrea/dev/uni/datasets/nsynth_small'
path_for_writing: str = '/home/andrea/dev/uni/datasets/nsynth'


def main():
    instruments: List[str] = os.listdir(path_to_data_to_augment)

    for instrument in instruments:
        path_to_samples: str = join(path_to_data_to_augment, instrument)
        sample_paths: List[str] = os.listdir(path_to_samples)

        print(f'\nProcessing {instrument}.')
        print(f'Num Samples: {len(sample_paths)}\n\n')

        for sample in sample_paths:
            print(f'Processing: {sample}')

            with ProcessPoolExecutor(max_workers=16) as process_pool_executor:
                process_pool_executor.submit(time_stretch_signal, path_to_samples, sample, instrument, path_for_writing)


if __name__ == '__main__':
    main()
