import os
import re
from utils.midi_note_table import MidiNoteTable


def main():
    # nsynth_midi_note_pattern: re.Pattern = re.compile(r'(?<=-)(.*?)(?=-)')
    midi_note_table = MidiNoteTable()

    print(midi_note_table.number_to_note(0))

if __name__ == '__main__':
    main()
