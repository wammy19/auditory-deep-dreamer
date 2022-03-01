from typing import List, Tuple

# Audio settings.
SAMPLE_RATE: int = 22_050
DELTA_TIME: float = 1.0  # Time of audio file in seconds.

# Mel Spec.
MEL_HOP_LEN: int = 512
NUM_MELS: int = 300
NUM_FFT: int = 2_048
MEL_TO_AUDIO_N_ITERATIONS: int = 32
MEL_WINDOW_LEN: int = 2_000

# Instrument table.
INSTRUMENT_ONTOLOGY: List[str] = [
    'bass',
    'brass',
    'flute',
    'guitar',
    'organ',
    'piano',
    'reed',
    'string',
    'synth',
    'vocal'
]

# Note Tables.
NOTE_TABLE: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SORTED_NOTE_TABLE: List[str] = sorted(NOTE_TABLE)

# Pre Processing.
TRIM_DB: int = 45

# Input data shape.
# Learning resources: https://stackoverflow.com/questions/62727244/what-is-the-second-number-in-the-mfccs-array/62733609#62733609
_y: int = 1 + SAMPLE_RATE // MEL_HOP_LEN
X_SHAPE: Tuple[int, int, int] = (NUM_MELS, _y, 1)
