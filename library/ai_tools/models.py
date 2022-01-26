from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, LayerNormalization, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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


def build_conv2d_example(N_CLASSES=2, sample_rate=consts.SAMPLE_RATE) -> Model:
    """
    :param N_CLASSES: Number of classes. Example: ['reed', 'string'] = 2
    :param sample_rate: Sample rate of .wav files.
    :return: A keras Sequential model.

    Builds a simple Conv2D network as an example.
    """

    input_shape = (sample_rate, 1)

    i = get_melspectrogram_layer(
        input_shape=input_shape,
        n_mels=consts.NUM_MELS,
        pad_end=True,
        n_fft=consts.NUM_FFT,
        win_length=consts.MEL_WINDOW_LEN,
        hop_length=consts.MEL_HOP_LEN,
        sample_rate=sample_rate,
        return_decibel=True,
        input_data_format='channels_last',
        output_data_format='channels_last'
    )

    x = LayerNormalization(axis=2, name='batch_norm')(i.output)

    # Block 1
    x = Conv2D(8, kernel_size=(7, 7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)

    # Block 2
    x = Conv2D(16, kernel_size=(5, 5), activation=relu, padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)

    # Block 3
    x = Conv2D(16, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)

    # Block 4
    x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)

    # Block 5
    x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_4')(x)
    x = Flatten(name='flatten')(x)

    x = Dropout(rate=0.2, name='dropout')(x)
    x = Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)

    output = Dense(N_CLASSES, activation=softmax, name='softmax')(x)

    model = Model(inputs=i.input, outputs=output, name='2d_convolution')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
