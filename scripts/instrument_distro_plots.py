# Creates a bar plot which informs us of the distribution of samples.

from collections import Counter
from typing import List

import matplotlib.pyplot as plt
from pandas import DataFrame

from ai_tools import DataGenerator
from ai_tools.helpers import decode_pitch
from utils.helpers import unix_url_substring_pattern


def main():
    data_gen = DataGenerator.from_path_to_audio(
        '/home/andrea/dev/uni/datasets/nsynth',
        number_of_samples_for_each_class=200_000
    )

    df: DataFrame = data_gen.get_data_frame
    num_paths: int = len(df.index)

    # Get instrument data.
    instruments: List[str] = [unix_url_substring_pattern.findall(df.loc[i]['path'])[0] for i in range(num_paths)]
    instruments_counter = Counter(instruments)

    # Instrument Bar Plot.
    plt.title('Instrument sample distribution after augmentation')
    plt.xlabel('Instrument families')
    plt.ylabel('Number of samples')
    plt.bar(instruments_counter.keys(), instruments_counter.values())
    plt.savefig('../plots/instrument_distro_original.png', bbox_inches='tight')

    # Get pitch data.
    # pitches: List[str] = [decode_pitch(df.loc[i]['pitch_label']) for i in range(num_paths)]
    # pitches_counter = Counter(pitches)

    plt.clf()

    # Pitch Bar Plot.
    # plt.title('Pitch distribution.')
    # plt.xlabel('Pitches')
    # plt.ylabel('Number of samples')
    # plt.bar(pitches_counter.keys(), pitches_counter.values())
    # plt.savefig('../plots/pitch_distro_post_time_stretching.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
