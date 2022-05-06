# Creates a bar plot which informs us of the distribution of samples.

from collections import Counter
from typing import List

import matplotlib.pyplot as plt
from pandas import DataFrame

from ai_tools import DataGenerator
from utils.helpers import unix_url_substring_pattern


def main():
    data_gen = DataGenerator.from_path_to_audio(
        '/storage/datasets/processed_dataset',
        number_of_samples_for_each_class=400_000
    )

    df: DataFrame = data_gen.get_data_frame
    num_paths: int = len(df.index)

    # Get instrument data.
    instruments: List[str] = [unix_url_substring_pattern.findall(df.loc[i]['path'])[0] for i in range(num_paths)]
    instruments_counter = Counter(instruments)

    print(instruments_counter.keys())
    print(instruments_counter.values())

    # Instrument Bar Plot.
    plt.title('Instrument sample distribution after augmentation')
    plt.xlabel('Instrument families')
    plt.ylabel('Number of samples')
    plt.bar(instruments_counter.keys(), instruments_counter.values())
    plt.savefig('../plots/instrument_distro_original.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
