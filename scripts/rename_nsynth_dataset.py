import os
from os.path import join
import re
from typing import List
from utils.helpers import midi_number_to_note

def rename_samples(path_to_audio_files, instrument: str) -> None:
    """
    :param path_to_audio_files:
    :param instrument:
    :return:
    """

    nsynth_midi_note_pattern: re.Pattern = re.compile(r'(?<=-)(.*?)(?=-)')
    samples: List[str] = os.listdir(path_to_audio_files)

    # Loop through all the samples and rename with a note rather than a midi number representing a note.
    for i, sample in enumerate(samples):
        path_to_sample: str = join(path_to_audio_files, sample)
        midi_note = int(nsynth_midi_note_pattern.findall(sample)[0])  # Number in range of 0-127
        note: str = midi_number_to_note(midi_note)
        new_name: str = f'{instrument}_{note}_{str(i).zfill(6)}.wav'
        new_name_path: str = join(path_to_audio_files, new_name)

        os.rename(path_to_sample, new_name_path)

def main():
    path_to_data_set: str = '../../data-sets/nsynth'
    ontology: List[str] = os.listdir(path_to_data_set)

    for instrument in ontology:
        path_to_instrument: str = join(path_to_data_set, instrument)
        rename_samples(path_to_instrument, instrument)


if __name__ == '__main__':
    main()
