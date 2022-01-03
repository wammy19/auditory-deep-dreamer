# Audio settings.
# General
SAMPLE_RATE: int = 22_050
DELTA_TIME: float = 1.0  # Time of audio file in seconds.

# Mel Spec.
MEL_HOP_LEN: int = 512
NUM_MELS: int = 128
NUM_FFT: int = 2_048
MEL_TO_AUDIO_N_ITERATIONS: int = 32
MEL_WINDOW_LEN: int = 400

# Pre Processing.
TRIM_DB: int = 45
