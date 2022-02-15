from ai_tools import DataGenerator
from collections import Counter
from pandas import DataFrame
import matplotlib.pyplot as plt
from typing import List
from utils.helpers import unix_url_substring_pattern


def main():
    data_gen = DataGenerator.from_path_to_audio('../../datasets/processed_dataset')
    df: DataFrame = data_gen.get_data_frame
    num_paths: int = len(df.index)
    instruments: List[str] = [unix_url_substring_pattern.findall(df.loc[i]['path'])[0] for i in range(num_paths)]
    instruments_counter = Counter(instruments)

    plt.bar(instruments_counter.keys(), instruments_counter.values())
    plt.savefig('instrument_distro.png')


if __name__ == '__main__':
    main()
