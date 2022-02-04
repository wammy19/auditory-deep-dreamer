from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
import os
from os.path import join
from shutil import move
from typing import List
from random import shuffle


def create_directories(path_to_dataset: str, ontology: List[str]) -> None:
    """
    :param: path_to_dataset:
    :return:

    Creates a 'validation', 'test', and 'train' directories in specified path.
    """

    os.makedirs(join(path_to_dataset, 'validation'), exist_ok=True)
    os.makedirs(join(path_to_dataset, 'test'), exist_ok=True)
    os.makedirs(join(path_to_dataset, 'train'), exist_ok=True)

    for instrument in ontology:
        os.makedirs(join(path_to_dataset, 'validation', instrument), exist_ok=True)
        os.makedirs(join(path_to_dataset, 'test', instrument), exist_ok=True)
        os.makedirs(join(path_to_dataset, 'train', instrument), exist_ok=True)


def move_files(path_to_dataset: str, instrument: str) -> None:
    """
    :param: path_to_dataset:
    :param: instrument:
    :return:

    Move files from their ontology into a validation, test, train structure.
    """

    instrument_paths: List[str] = os.listdir(join(path_to_dataset, instrument))
    shuffle(instrument_paths)

    number_of_paths: int = len(instrument_paths)
    test_split_amount: int = number_of_paths // 10  # 10% of entire data set for testing and validation.
    test_split: List[str] = instrument_paths[:test_split_amount].copy()
    validation_split: List[str] = instrument_paths[test_split_amount:test_split_amount * 2].copy()
    train_split: List[str] = instrument_paths[test_split_amount * 3:].copy()

    del instrument_paths  # Clear memory.

    # Move test data.
    for test, validation in zip(test_split, validation_split):
        try:
            move(join(path_to_dataset, instrument, test), join(path_to_dataset, 'test', instrument))
            move(join(path_to_dataset, instrument, validation), join(path_to_dataset, 'validation', instrument))

        except FileNotFoundError as err:
            print(err)

    # Move training data.
    for train in train_split:
        try:
            move(join(path_to_dataset, instrument, train), join(path_to_dataset, 'train', instrument))

        except FileNotFoundError as err:
            print(err)


def main() -> None:
    path_to_dataset: str = '../../data-sets/processed_nsynth'
    ontology: List[str] = sorted(os.listdir(path_to_dataset))
    num_workers: int = 3

    if 'test' in ontology:
        ontology.remove('validation')
        ontology.remove('test')
        ontology.remove('train')

    create_directories(path_to_dataset, ontology)

    for instrument in ontology:
        move_files(path_to_dataset, instrument)


if __name__ == '__main__':
    main()
