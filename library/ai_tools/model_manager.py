import os
from csv import DictWriter
from typing import List, Optional, Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, LayerNormalization, MaxPooling2D
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
            model_name: Optional[str] = None,
            num_conv_blocks: int = 10,
            num_dense_blocks: int = 20,
            conv_drop_out: bool = False
    ) -> None:
        """
        :param num_conv_blocks:
        """

        if model_name is None:
            pass

        self.model_name = model_name
        self.num_conv_blocks = num_conv_blocks
        self.num_dense_blocks = num_dense_blocks
        self.conv_drop_out = conv_drop_out


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

        for _ in range(self.num_conv_blocks):
            x = Conv2D(128, kernel_size=(5, 5), activation=relu, padding='same')(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            if self.conv_drop_out:
                x = Dropout(rate=0.2)(x)

        x = Flatten(name='flatten')(x)

        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=relu, activity_regularizer=l2(0.001))(x)

        output = Dense(num_classes, activation=softmax, name='softmax')(x)
        model = Model(inputs=input_layer, outputs=output, name='2d_convolution')

        model.compile(
            optimizer='adam',
            loss=categorical_crossentropy,
            metrics=['accuracy']
        )

        return model


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
