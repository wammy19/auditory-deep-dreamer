import numpy as np
from tensorflow.keras.models import Model, load_model
from ai_tools import ModelFilterAmplifier
from librosa import load
from utils.audio_tools import create_audio_player
from utils.displays import display_mel_spectrogram
from soundfile import write
from os.path import join
from os import makedirs
import utils.constants as consts
from matplotlib import pyplot as plt
from librosa.display import specshow

PATH_TO_MODEL: str = '../models/model_3/epoch-96.pb'
PATH_TO_AUDIO: str = '../long-audio/debussy.wav'
OUTPUT_DIR: str = '../feature_visualization'

makedirs(OUTPUT_DIR, exist_ok=True)
makedirs(join(OUTPUT_DIR, 'mel_specs'), exist_ok=True)

# Load model and sample.
model: Model = load_model(PATH_TO_MODEL)
sample: np.ndarray = load(PATH_TO_AUDIO)[0]

# Write audio file.
write(join(OUTPUT_DIR, 'input') + '.wav', sample, samplerate=consts.SAMPLE_RATE)

model_filter_amplifier = ModelFilterAmplifier(model, 2)

# Loop over all neurons and save their output.
for neuron_i in range(model_filter_amplifier.max_num_neurons):

    print(f'Processing neuron: {neuron_i}')

    audio_signal, mel_spec = model_filter_amplifier(
        sample,
        filter_index=neuron_i,
        iterations=1,
        filter_amount=0.06,  # Mild high cut. 0 is no filtering, 1 is completely closed filter.
    )  # type: np.ndarray, np.ndarray

    neuron_name: str = f'neuron_{neuron_i}'

    # Write audio file.
    write(join(OUTPUT_DIR, neuron_name) + '.wav', audio_signal, samplerate=consts.SAMPLE_RATE)

    # Write mel spectrogram to file.
    specshow(mel_spec, sr=consts.SAMPLE_RATE)
    plt.savefig(join(OUTPUT_DIR, 'mel_specs', neuron_name + '.png'))
