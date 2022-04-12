import os
import re
from os.path import join
from typing import List


def main():
    note_pattern: re.Pattern = re.compile(r'(?<=_)(.*?)(?=_)')
    path_to_dataset: str = '../../datasets/processed_philharmonia_dataset'
    ontology: List[str] = os.listdir(path_to_dataset)

    for instrument in ontology:
        path_to_samples: str = join(path_to_dataset, instrument)

        for i, sample in enumerate(os.listdir(path_to_samples)):
            sample_note: str = note_pattern.findall(sample)[0]
            path_to_sample: str = join(path_to_samples, sample)

            # If note is sharp.
            if 's' in sample_note:
                new_sample_name: str = f'phil_{instrument}_{sample_note[0]}#_{str(i).zfill(6)}.wav'
            else:
                new_sample_name: str = f'phil_{instrument}_{sample_note[0]}_{str(i).zfill(6)}.wav'

            new_name_path: str = join(path_to_samples, new_sample_name)
            os.rename(path_to_sample, new_name_path)


if __name__ == '__main__':
    main()
