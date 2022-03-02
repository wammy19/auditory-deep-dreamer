from typing import Tuple

from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LayerNormalization, MaxPooling2D
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import utils.constants as consts


def build_simple_cnn() -> Sequential:
    """
    :return: A compiled tensorflow.keras.Model ready for fitting.

    An example model for testing purposes.
    """

    # Shape: (num_samples, sample_rate, channels)
    input_shape: Tuple[int, int] = (consts.SAMPLE_RATE, 1)  # Mono audio so 1 channel.

    # Mel-spectrogram encoding layer.
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
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and a binary output for final layer.
    model.add(Flatten())
    model.add(Dense(1, activation=sigmoid))

    model.compile(
        loss=binary_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model


def build_conv2d_example(N_CLASSES: int = 10, input_shape: Tuple[int, int, int] = consts.X_SHAPE) -> Model:
    """
    :param N_CLASSES: Number of classes. Example: ['reed', 'string'] = 2
    :param input_shape: Input shape of the data not including batch size.
    :return: A keras Sequential model.

    Builds a simple Conv2D network as an example.
    """

    input_layer = Input(shape=input_shape)

    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    # Block 1
    x = Conv2D(128, kernel_size=(7, 7), activation=relu, padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)

    # Block 2
    x = Conv2D(128, kernel_size=(5, 5), activation=relu, padding='same', name='conv2d_relu_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)

    # Block 3
    x = Conv2D(64, kernel_size=(5, 5), activation=relu, padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)

    # Block 4
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)

    # Block 5
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_5')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_5')(x)

    # Block 6
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_6')(x)

    # Block 7
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_7')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_7')(x)

    # Block 7
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_8')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_8')(x)

    # Block 7
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_9')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_9')(x)

    # Block 7
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_10')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_10')(x)

    x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_11')(x)
    x = Flatten(name='flatten')(x)

    x = Dropout(rate=0.2, name='dropout')(x)
    x = Dense(64, activation=relu, activity_regularizer=l2(0.001), name='dense_1')(x)

    output = Dense(N_CLASSES, activation=softmax, name='softmax')(x)
    model = Model(inputs=input_layer, outputs=output, name='2d_convolution')

    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    return model
