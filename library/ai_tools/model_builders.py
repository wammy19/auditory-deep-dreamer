from typing import Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, LayerNormalization, \
    MaxPooling2D, SeparableConv2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2

import utils.constants as consts


def simple_model_for_test() -> Model:
    """
    :return:
    """

    input_layer = Input(shape=consts.X_SHAPE)

    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    # Block 1
    x = Conv2D(32, kernel_size=(7, 7), activation=relu, padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)

    x = Flatten(name='flatten')(x)

    x = Dropout(rate=0.2, name='dropout')(x)
    x = Dense(32, activation=relu, activity_regularizer=l2(0.001), name='dense_1')(x)

    output = Dense(10, activation=softmax, name='softmax')(x)
    model = Model(inputs=input_layer, outputs=output, name='2d_convolution')

    model.compile(
        optimizer='adam',
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


def sequential_conv_2d(
        num_conv_blocks: int = 1,
        num_filters: int = 16,
        num_dense_layers: int = 0,
        dense_layer_size: int = 32,
        use_separable_conv_layer: bool = False,
        use_regularization: bool = False,
        use_dropout_dense_layers: bool = False,
        use_dropout_conv_blocks: bool = False,
        dense_dropout_amount: float = 0.5,
        conv_dropout_amount: float = 0.1,
        regularization_amount: float = 0.001,
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = consts.X_SHAPE,
) -> Model:
    input_layer = Input(shape=input_shape)

    # Normalize data.
    x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

    # Set kernel size for conv layer. This will decrease over every layer if there are more than 3 blocks.
    if num_conv_blocks >= 3:
        kernel_size = 7
    else:
        kernel_size = 3

    # Conv blocks.
    for block_num in range(num_conv_blocks):

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

        # Batch normalization is added for each block as suggested in "Deep Learning with Python" by Francois Chollet.
        # "BatchNormalization is used liberally in many of the advanced convent architectures." [8]
        x = BatchNormalization(name=f'batch_norm_{block_num}')(x)

        # Add dropout.
        if use_dropout_conv_blocks:
            x = Dropout(conv_dropout_amount, name=f'conv_dropout{block_num}')(x)

        # Decrease kernel size. Pattern:
        # Layer 1 kernel size = 7
        # Layer 2 kernel size = 5
        # Layer 3 and greater kernel size = 3
        if num_conv_blocks >= 3:
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

    # Block 8
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_8')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_8')(x)

    # Block 9
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_9')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_9')(x)

    # Block 10
    x = Conv2D(64, kernel_size=(3, 3), activation=relu, padding='same', name='conv2d_relu_10')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_10')(x)

    # Block 11
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
