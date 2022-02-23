"""
Creates a data frame out of the dataset and saves it to a CSV file.

Example of CSV file:

index, path_to_data, instrument_label (one-hot-encoded), pitch_label (one-hot-encoded)
0,/path/to/data/bass/bass_A#_000006_segment_0.wav,[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
1,/path/to/data/bass/bass_A#_000007_segment_0.wav,[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
2,/path/to/data/bass/bass_A#_000007_segment_1.wav,[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
3,/path/to/data/bass/bass_A#_000007_segment_2.wav,[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.],[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
"""

from os.path import join

from pandas import DataFrame

from ai_tools.helpers import create_data_frame_from_path
from utils.helpers import read_yaml_config


def main():
    yaml_config: dict = read_yaml_config()

    # Config
    csv_save_path: str = '../datasets'
    path_to_dataset: str = yaml_config['path_to_dataset']

    df: DataFrame = create_data_frame_from_path(path_to_dataset)
    df.to_csv(join(csv_save_path, 'all-dataset-paths.csv'))


if __name__ == '__main__':
    main()
