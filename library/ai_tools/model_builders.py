from typing import Tuple

from kapre.composed import get_melspectrogram_layer
from tensorflow.keras import Input, Sequential
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, LayerNormalization, \
    MaxPooling2D, SeparableConv2D, SpatialDropout2D
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import utils.constants as consts
from utils.constants import X_SHAPE


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

    input_layer = Input(input_shape=input_shape)

    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    # Block 1
    x = Conv2D(8, kernel_size=(7, 7), activation=relu, padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)

    # Block 2
    x = Conv2D(16, kernel_size=(5, 5), activation=relu, padding='same', name='conv2d_relu_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)

    # Block 3
    x = Conv2D(16, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)

    # Block 4
    x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)

    # Block 5
    x = Conv2D(32, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_5')(x)
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


def dynamic_conv2d_model(
        num_conv_block: int,
        num_filters: int,
        num_dense_layers: int,
        dense_layer_size: int,
        use_separable_conv_layer: bool,
        use_regularization: bool,
        use_dropout_dense_layers: bool,
        use_dropout_conv_blocks: bool,
        dense_dropout_amount: float,
        conv_dropout_amount: float,
        regularization_amount: float,
        num_classes: int,
        input_shape: Tuple[int, int, int] = X_SHAPE,
) -> Model:
    """
    :param num_conv_block: Number of Conv2D layers, this includes max pooling after each Conv2D layer.
    :param num_filters:
    :param num_dense_layers:
    :param dense_layer_size:
    :param use_separable_conv_layer:
    :param use_regularization:
    :param use_dropout_dense_layers:
    :param use_dropout_conv_blocks:
    :param dense_dropout_amount:
    :param conv_dropout_amount:
    :param regularization_amount:
    :param num_classes: Number of classes. Example: ['reed', 'string', 'guitar'] = 3
    :param input_shape: Input shape of the data not including batch size.
    :return: A tensorflow.keras.models.Model object ready for training.
    Constructs a Convolutional Neural Network.
    """

    print(type(num_conv_block))
    print(num_conv_block)
    print('\n\n')

    input_layer = Input(shape=input_shape)

    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    # Set kernel size for conv layer. This will decrease over every layer if there are more than 3 blocks.
    if num_conv_block >= 3:
        kernel_size = 7
    else:
        kernel_size = 3

    # Conv blocks.
    for block_num in range(num_conv_block):

        if use_separable_conv_layer:
            x = SeparableConv2D(
                num_filters,
                kernel_size=(kernel_size, kernel_size),
                activation=relu,
                padding='same',
                name=f'conv_block_{block_num}'
            )(x)

        else:
            x = Conv2D(
                num_filters,
                kernel_size=(kernel_size, kernel_size),
                activation=relu,
                padding='same',
                name=f'conv_block_{block_num}'
            )(x)

        x = MaxPooling2D(pool_size=(2, 2), padding='same', name=f'pooling_{block_num}')(x)

        # Batch normalization is added for each block as suggested in "Deep Learning with Python"
        # by Francois Chollet.
        # "BatchNormalization is used liberally in many of the advanced convent architectures." [8]
        x = BatchNormalization(name=f'batch_norm_{block_num}')(x)

        # Add dropout.
        if use_dropout_conv_blocks:
            x = SpatialDropout2D(conv_dropout_amount, name=f'conv_dropout{block_num}')(x)

        # Decrease kernel size. Pattern:
        # Layer 1 kernel size = 7
        # Layer 2 kernel size = 5
        # Layer 3 and greater kernel size = 3
        if num_conv_block >= 3:
            if block_num == 1:
                kernel_size = 5

            elif block_num > 2:
                kernel_size = 3

    x = Flatten(name='flatten')(x)

    # Dense layers.
    for dense_layer_num in range(num_dense_layers):
        if use_regularization:
            x = Dense(
                dense_layer_size,
                activation=relu,
                activity_regularizer=l2(regularization_amount),
                name=f'dense_{dense_layer_num}'
            )(x)

        else:
            x = Dense(
                dense_layer_size,
                activation=relu,
                name=f'dense_{dense_layer_num}'
            )(x)

        if use_dropout_dense_layers:
            x = Dropout(dense_dropout_amount, name=f'dense_dropout_{dense_layer_num}')(x)

    # Final softmax layer
    output = Dense(num_classes, activation=softmax, name='soft_max_output')(x)

    # Create model.
    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


def vgg_like_model(
        dropout_amount: float = 0.1,
        learning_rate: float = 0.001,
        input_shape: Tuple[int, int, int] = X_SHAPE,
        num_classes: int = 10
) -> Model:
    """
    :param dropout_amount:
    :param learning_rate:
    :param input_shape:
    :param num_classes:
    :return:
    """

    input_layer = Input(shape=input_shape)

    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    kernel_size: int = 3

    # Conv block 1
    x = Conv2D(
        64,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        64,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        128,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        256,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        256,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        256,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        256,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        512,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Conv2D(
        512,
        kernel_size=(kernel_size, kernel_size),
        activation=relu,
        padding='same',
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_amount)(x)

    x = Flatten()(x)

    Dense(128, activation=relu)(x)

    # Final softmax layer
    output = Dense(num_classes, activation=softmax, name='soft_max_output')(x)

    # Create model.
    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    return model