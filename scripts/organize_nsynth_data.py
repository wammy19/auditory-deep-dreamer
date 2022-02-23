"""
Helper script for organizing Google's nSynth dataset into a different file structure from what is downloaded.
"""

import os
import re
import shutil

from tqdm import tqdm


def main():
    patter_for_instrument: re.Pattern = re.compile(r'^.*?(?=_)')
    path_to_dataset: str = '../data-sets/nsynth/all-files'

    for file in tqdm(os.listdir(path_to_dataset)):

        if os.path.isdir(os.path.join(path_to_dataset, file)):  # Ignore directories.
            continue

        instrument: str = re.findall(patter_for_instrument, file)[0]
        instrument_dir: str = os.path.join(path_to_dataset, instrument)

        if os.path.isdir(instrument_dir) is False:
            os.makedirs(instrument_dir)

        shutil.move(os.path.join(path_to_dataset, file), instrument_dir)  # Move file into appropriate folder.


if __name__ == '__main__':
    main()
