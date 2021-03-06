from os import makedirs
from os.path import join

import numpy as np
from librosa import load
from librosa.display import specshow
from matplotlib import pyplot as plt
from soundfile import write
from tensorflow.keras.models import Model, load_model

import utils.constants as consts
from ai_tools import ModelFilterAmplifier

PATH_TO_MODEL: str = '../runs/models/model_3/epoch-96.pb'
PATH_TO_AUDIO: str = '../long-audio/casio_1.wav'
OUTPUT_DIR: str = '../feature_visualization'

makedirs(OUTPUT_DIR, exist_ok=True)
makedirs(join(OUTPUT_DIR, 'mel_specs'), exist_ok=True)

model: Model = load_model(PATH_TO_MODEL)
sample: np.ndarray = load(PATH_TO_AUDIO)[0]

# Write audio file.
write(join(OUTPUT_DIR, 'input') + '.wav', sample, samplerate=consts.SAMPLE_RATE)

model_filter_amplifier = ModelFilterAmplifier(model, 2)

# Loop over all neurons and save their output.
for neuron_i in range(model_filter_amplifier.max_num_neurons):
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
