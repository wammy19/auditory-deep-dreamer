from typing import List

class MidiNoteTable:
    """
    A table of notes that can be queried using a midi number.
    source: https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
    """

    _notes: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    _note_in_octave = len(_notes)

    def number_to_note(self, number: int) -> str:
        """
        :param: number:
        :return:
        """

        if number > 127 or number < 0:
            raise ValueError("Number must be in the range of 0-127")

        note = self._notes[number % self._note_in_octave]
        return note