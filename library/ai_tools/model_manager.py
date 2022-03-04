import os
from csv import DictWriter
from typing import List, Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LayerNormalization, MaxPooling2D, BatchNormalization, SeparableConv2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2

from utils.constants import X_SHAPE


class ModelManager:
    """
    A data structure that holds the settings for creating a model.
    """


    # =================================================================================================================
    # ---------------------------------------------- Class Constructors -----------------------------------------------
    # =================================================================================================================
    def __init__(
            self,
            model_name: str,
            num_conv_block=1,
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
    ):
        self.model_name: str = model_name
        self.num_conv_block: int = num_conv_block
        self.num_filters: int = num_filters
        self.num_dense_layers: int = num_dense_layers
        self.dense_layer_size: int = dense_layer_size
        self.use_separable_conv_layer: bool = use_separable_conv_layer
        self.use_regularization: bool = use_regularization
        self.use_dropout_dense_layers: bool = use_dropout_dense_layers
        self.use_dropout_conv_blocks: bool = use_dropout_conv_blocks
        self.dense_dropout_amount: float = dense_dropout_amount
        self.conv_dropout_amount: float = conv_dropout_amount
        self.regularization_amount: float = regularization_amount


    # =================================================================================================================
    # ----------------------------------------------- Public functions ------------------------------------------------
    # =================================================================================================================

    def build_model(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = X_SHAPE) -> Model:
        """
        :param num_classes: Number of classes. Example: ['reed', 'string', 'guitar'] = 3
        :param input_shape: Input shape of the data not including batch size.
        :return: A tensorflow.keras.models.Model object ready for training.

        Constructs a model with the stored settings and returns a model.
        """

        input_layer = Input(shape=input_shape)

        x = LayerNormalization(axis=2, name='batch_norm')(input_layer)

        # Set kernel size for conv layer. This will decrease over every layer if there are more than 3 blocks.
        if self.num_conv_block >= 3:
            kernel_size = 7
        else:
            kernel_size = 3

        # Conv blocks.
        for block_num in range(self.num_conv_block):

            if self.use_separable_conv_layer:
                x = SeparableConv2D(
                    self.num_filters,
                    kernel_size=(kernel_size, kernel_size),
                    activation=relu,
                    padding='same',
                    name=f'conv_block_{block_num}'
                )(x)

            else:
                x = Conv2D(
                    self.num_filters,
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
            if self.use_dropout_conv_blocks:
                x = Dropout(self.conv_dropout_amount, name=f'conv_dropout{block_num}')(x)

            # Decrease kernel size. Pattern:
            # Layer 1 kernel size = 7
            # Layer 2 kernel size = 5
            # Layer 3 and greater kernel size = 3
            if self.num_conv_block >= 3:
                if block_num == 1:
                    kernel_size = 5

                elif block_num > 2:
                    kernel_size = 3

        x = Flatten(name='flatten')(x)

        # Dense layers.
        for dense_layer_num in range(self.num_dense_layers):
            if self.use_regularization:
                x = Dense(
                    self.dense_layer_size,
                    activation=relu,
                    activity_regularizer=l2(self.regularization_amount),
                    name=f'dense_{dense_layer_num}'
                )(x)

            else:
                x = Dense(
                    self.dense_layer_size,
                    activation=relu,
                    name=f'dense_{dense_layer_num}'
                )(x)

            if self.use_dropout_dense_layers:
                x = Dropout(self.dense_dropout_amount, name=f'dense_dropout_{dense_layer_num}')(x)

        # Final softmax layer
        output = Dense(num_classes, activation=softmax, name='soft_max_output')(x)

        # Create model.
        _model = Model(inputs=input_layer, outputs=output)

        _model.compile(
            optimizer='adam',
            loss=categorical_crossentropy,
            metrics=['accuracy']
        )

        return _model


    def save_model_settings_to_csv(self, path_to_csv: str = '../model_settings.csv') -> None:
        """
        :return:

        Appends the model settings to a csv file.
        """

        csv_headers: List[str] = []

        for key, value in self.__dict__.items():  # Gather column headers.
            csv_headers.append(key)

        with open(path_to_csv, 'a', newline='') as file_handler:
            dict_writer = DictWriter(file_handler, fieldnames=csv_headers)

            # Write column headers if the file is new.
            if os.stat(path_to_csv).st_size == 0:
                dict_writer.writeheader()

            dict_writer.writerow(self.__dict__)
