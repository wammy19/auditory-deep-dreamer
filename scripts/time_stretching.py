from concurrent.futures import ProcessPoolExecutor
from librosa import load
from librosa.effects import time_stretch
import numpy as np
import os
from os.path import join
import re
import soundfile as sf  # https://pysoundfile.readthedocs.io/en/latest/index.html#soundfile.write
from typing import List
import utils.constants as consts


# Settings.
path_to_data_to_augment: str = '/home/andrea/dev/uni/datasets/nsynth_small'
path_for_writing: str = '/home/andrea/dev/uni/datasets/nsynth'
file_name_pattern: re.Pattern = re.compile(r'^.+?(?=\.)')


def augment_data(path_to_samples: str, sample: str, instrument):
    loaded_sample: np.ndarray = load(join(path_to_samples, sample), sr=consts.SAMPLE_RATE, mono=True)[0]
    stretched_sample: np.ndarray = time_stretch(loaded_sample, rate=0.5)
    file_name: str = f'{file_name_pattern.findall(sample)[0]}_stretched.wav'

    sf.write(join(path_for_writing, instrument, file_name), stretched_sample, consts.SAMPLE_RATE)


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
                process_pool_executor.submit(augment_data, path_to_samples, sample, instrument)


if __name__ == '__main__':
    main()
