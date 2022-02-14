from ai_tools.encoders import get_instrument_encodings, get_pitch_encodings
import numpy as np
import os
from os.path import join
from pandas import DataFrame
from typing import List
from utils.helpers import read_yaml_config, get_paths_to_wav_files


def main():
    # Config
    csv_save_path: str = '../datasets'
    path_to_dataset: str = yaml_config['path_to_dataset']

    wav_paths: List[str] = get_paths_to_wav_files(path_to_dataset)
    instrument_classes: List[str] = sorted(os.listdir(path_to_dataset))  # Example: ['string', 'reed']

    # One hot encoded labels.
    instrument_labels: np.ndarray = get_instrument_encodings(wav_paths, instrument_classes)
    pitch_labels: np.ndarray = get_pitch_encodings(wav_paths)

    df = DataFrame.from_dict(
        {
            'path': wav_paths,
            'instrument_label': [label for label in instrument_labels],
            'pitch_label': [label for label in pitch_labels]
        }
    )

    df.to_csv(join(csv_save_path, 'all_data_set_paths.csv'))


if __name__ == '__main__':
    global yaml_config
    yaml_config: dict = read_yaml_config()

    main()
