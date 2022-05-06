import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from keras.engine.keras_tensor import KerasTensor
from librosa import power_to_db
from librosa.display import specshow
from librosa.feature.inverse import mel_to_audio
from soundfile import write
from tensorflow.keras.models import Model, load_model

import utils.constants as consts
from utils.audio_tools import load_and_convert_audio_into_mel_spectrogram
from utils.constants import X_SHAPE

PATH_TO_MODEL: str = '../models/model_0/epoch-26.pb'
PATH_TO_AUDIO: str = '../media/audio/string_a#.wav'
OUTPUT_DIR: str = '../layer_outputs'
NUM_LAYERS: int = 44
NUM_WORKERS: int = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_neuron_output(layer_activation, neuron_i, path_to_out_dir, path_to_mel_specs):
    """
    :param layer_activation:
    :param neuron_i:
    :param path_to_out_dir:
    :param path_to_mel_specs:
    :return:
    """

    print(f'Processing {neuron_i}')

    mel_spec: np.ndarray = layer_activation[0, :, :, neuron_i]
    pad_left: np.ndarray = X_SHAPE[0] - mel_spec.shape[0]
    pad_right: np.ndarray = X_SHAPE[1] - mel_spec.shape[1]
    mel_spec = np.pad(mel_spec, ((0, pad_left), (0, pad_right)), 'minimum')

    signal: np.ndarray = mel_to_audio(
        mel_spec,
        sr=consts.SAMPLE_RATE,
        n_iter=consts.MEL_TO_AUDIO_N_ITERATIONS
    )

    neuron_name: str = f'neuron_{neuron_i}'
    write(join(path_to_out_dir, neuron_name) + '.wav', signal, consts.SAMPLE_RATE)

    mel_spec_log: np.ndarray = power_to_db(mel_spec, ref=np.max)
    specshow(mel_spec_log, sr=consts.SAMPLE_RATE)
    plt.savefig(join(path_to_mel_specs, neuron_name + '.png'))


def main():
    input_data: np.ndarray = load_and_convert_audio_into_mel_spectrogram(PATH_TO_AUDIO, True, 1.0)

    print('Loading model.')
    # Create model for feature visualization.
    model: Model = load_model(PATH_TO_MODEL)
    layer_output: List[KerasTensor] = [layer.output for layer in model.layers[:NUM_LAYERS]]
    activation_model = Model(inputs=model.input, outputs=layer_output)

    print('Gathering layers.')

    # Collect indicis to model's conv block layers.
    layers = activation_model.layers
    conv_block_indicis: List[int] = []

    for i, layer in enumerate(layers):
        if 'conv_block' in layer.name:
            conv_block_indicis.append(i)

    activations: List[np.ndarray] = activation_model.predict(input_data)

    print('Looping over layers.')

    # Loop over all conv blocks.
    for layer_i in conv_block_indicis:
        layer_activation: np.ndarray = activations[layer_i]
        layer_name: str = f'conv_block_{layer_i}'

        path_to_out_dir: str = join(OUTPUT_DIR, layer_name)
        path_to_mel_specs: str = join(path_to_out_dir, 'mel_specs')

        os.makedirs(path_to_out_dir, exist_ok=True)
        os.makedirs(path_to_mel_specs, exist_ok=True)

        # Loop over each neuron and save audio.
        for neuron_i in range(layer_activation.shape[-1] - 1):
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as process_pool_executor:
                process_pool_executor.submit(
                    save_neuron_output,
                    layer_activation,
                    neuron_i,
                    path_to_out_dir,
                    path_to_mel_specs
                )


if __name__ == '__main__':
    main()
