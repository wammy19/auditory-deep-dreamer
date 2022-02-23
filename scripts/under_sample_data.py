from pprint import pprint

from pandas import DataFrame

from ai_tools.helpers import create_data_frame_from_path

path_to_data_set: str = '/home/andrea/dev/uni/datasets/processed_dataset'


def main():
    wav_paths: DataFrame = create_data_frame_from_path(path_to_data_set)
    print(wav_paths.head())


if __name__ == '__main__':
    main()
