import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from random import shuffle
from shutil import move
from typing import List

test_set_split: int = 5  # Percent.


def create_directories(path_to_dataset: str, ontology: List[str]) -> None:
    """
    :param path_to_dataset:
    :param ontology:
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


def check_if_lists_are_similar(list_1: List[str], list_2: List) -> bool:
    """
    :param list_1:
    :param list_2:
    :return:

    Returns True if the lists are similar. The lists are similar if at least 1 element is shared by the lists.
    """

    for element_1 in list_1:
        for element_2 in list_2:
            if element_1 == element_2:
                return True

    return False


def move_files(path_to_dataset: str, instrument: str) -> None:
    """
    :param path_to_dataset:
    :param instrument:
    :return:

    Move files from their ontology into a validation, test, train structure.
    """

    instrument_paths: List[str] = os.listdir(join(path_to_dataset, instrument))
    shuffle(instrument_paths)

    number_of_paths: int = len(instrument_paths)
    test_split_amount: int = number_of_paths // test_set_split

    # Splits
    test_split: List[str] = instrument_paths[:test_split_amount].copy()
    validation_split: List[str] = instrument_paths[test_split_amount:test_split_amount * 2].copy()
    train_split: List[str] = instrument_paths[test_split_amount * 2:].copy()

    # Run tests to validate data.
    assert len(test_split) == test_split_amount
    assert len(validation_split) == test_split_amount
    assert len(train_split) == number_of_paths - (len(test_split) + len(validation_split))
    assert number_of_paths == len(test_split) + len(validation_split) + len(train_split)
    assert test_split != validation_split

    # Check if any paths are bleeding into the splits.
    assert check_if_lists_are_similar(test_split, validation_split) is False
    assert check_if_lists_are_similar(train_split, validation_split) is False
    assert check_if_lists_are_similar(train_split, test_split) is False

    del instrument_paths

    path_to_sample: str = join(path_to_dataset, instrument)

    # Move test data.
    for test, validation in zip(test_split, validation_split):
        try:
            move(join(path_to_sample, test), join(path_to_dataset, 'test', instrument))
            move(join(path_to_sample, validation), join(path_to_dataset, 'validation', instrument))

        except FileNotFoundError as err:
            print(err)

    # Move training data.
    for train in train_split:
        try:
            move(join(path_to_sample, train), join(path_to_dataset, 'train', instrument))

        except FileNotFoundError as err:
            print(err)


def main() -> None:
    path_to_dataset: str = '../../datasets/philharmonia_split'
    ontology: List[str] = sorted(os.listdir(path_to_dataset))

    if 'test' in ontology:
        ontology.remove('validation')
        ontology.remove('test')
        ontology.remove('train')

    create_directories(path_to_dataset, ontology)

    with ProcessPoolExecutor(max_workers=len(ontology)) as process_executor:
        for instrument in ontology:
            process_executor.submit(move_files, path_to_dataset, instrument)

    for instrument in ontology:
        os.rmdir(join(path_to_dataset, instrument))


if __name__ == '__main__':
    main()
