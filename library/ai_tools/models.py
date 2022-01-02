from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from typing import Tuple
import utils.constants as consts


def build_simple_cnn() -> Sequential:
    """
    :return: A compiled tensorflow.keras.Model ready for fitting.

    An example model for testing purposes.
    """

    # Shape: (num_samples, sample_rate, channels)
    input_shape: Tuple[int, int] = (consts.SAMPLE_RATE, 1)  # Mono audio so 1 channel.

    # First layer transforms raw audio into a mel-spectrogram in GPU.
    mel_layer: Sequential = get_melspectrogram_layer(
        input_shape=input_shape,
        n_fft=consts.NUM_FFT,
        hop_length=consts.MEL_HOP_LEN,
        sample_rate=consts.SAMPLE_RATE,
        return_decibel=True,
        input_data_format='channels_last',
        output_data_format='channels_last'
    )

    model = Sequential()

    # Mel transformation.
    model.add(mel_layer)

    # Conv block 1.
    model.add(Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        activation=relu
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and a binary output for final layer.
    model.add(Flatten())
    model.add(Dense(1, activation=sigmoid))

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model
