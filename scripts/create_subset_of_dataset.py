"""Script that creates a smaller version of the entire dataset for development on laptop."""

import os
from os.path import join
from random import shuffle
from shutil import copy
from typing import List

numb_samples_per_instrument: int = 100
path_to_dataset: str = '../../datasets/processed_dataset'
path_to_subset: str = '../../datasets/processed_dataset_small'


def main():
    ontology: List[str] = os.listdir(path_to_dataset)

    for instrument in ontology:
        path_to_samples: str = join(path_to_dataset, instrument)
        samples: List[str] = os.listdir(path_to_samples)

        shuffle(samples)

        for i in range(numb_samples_per_instrument):
            full_path_to_sample: str = join(path_to_samples, samples[i])

            copy(full_path_to_sample, join(path_to_subset, instrument))


if __name__ == '__main__':
    main()
