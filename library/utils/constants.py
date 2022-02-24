from typing import List

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
