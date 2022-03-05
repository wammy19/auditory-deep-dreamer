import os
from csv import DictWriter
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

from aim.keras import AimCallback
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, LayerNormalization, \
    MaxPooling2D, SeparableConv2D, SpatialDropout2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.models import Model

from ai_tools import DataGenerator
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
            path_to_csv_logs: str = './model_settings.csv',
            model_checkpoint_dir: str = './models',
            aim_logs_dir: str = './aim',
            training_batch_size: int = 32,
            num_training_epochs: int = 200,
            model: Optional[Model] = None,
    ):
        """
        :param path_to_csv_logs:
        :param model_checkpoint_dir:
        :param aim_logs_dir:
        :param training_batch_size:
        :param num_training_epochs:
        :param model:
        """

        # Logs paths.
        self._path_to_csv_logs: str = path_to_csv_logs
        self._model_checkpoint_dir: str = model_checkpoint_dir
        self._aim_logs_dir: str = aim_logs_dir

        # Initialize model ID.
        self._model_ID: int = len(os.listdir(self._model_checkpoint_dir))

        # Training settings.
        self.num_epochs: int = num_training_epochs
        self._batch_size: int = training_batch_size
        self._current_model: Optional[Model] = model


    # =================================================================================================================
    # ----------------------------------------------- Public functions ------------------------------------------------
    # =================================================================================================================

    def build_model(
            self,
            num_conv_block: int = 1,
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

        model_settings: Dict[str, any] = locals()  # Store model settings.
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

            # Batch normalization is added for each block as suggested in "Deep Learning with Python" by Francois Chollet.
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

        self.current_model = model
        self._save_model_settings_to_csv(model_settings)  # Save the variables that constructed this model.
        self._model_ID += 1  # Increment model ID for the next model.

        return model


    def train_and_evaluate_model(
            self,
            train_generator: DataGenerator,
            validation_generator: DataGenerator,
            model: Optional[Model] = None,
    ) -> History:
        """
        :param model: A tensorflow.keras.models.Model ready for training.
        :param train_generator: A DataGenerator holding the training dataset.
        :param validation_generator: A DataGenerator holding the validation dataset.
        :return:
        """

        if model is None:
            model = self.current_model

        # Callbacks.
        aim_callback = AimCallback(repo=self._aim_logs_dir, experiment='model_test')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=False)
        checkpoint = ModelCheckpoint(
            join(join(self._model_checkpoint_dir, f'model_{self._model_ID}'), 'epoch-{epoch:02d}.pb'),
            monitor='val_accuracy',
            verbose=False,
            save_weights_only=False,
            save_best_only=True,
            mode='max',
        )

        # Train model.
        history: History = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator.get_data_frame.index) // self._batch_size,
            epochs=self.num_epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator.get_data_frame.index) // self._batch_size,
            batch_size=self._batch_size,
            callbacks=[
                aim_callback,
                early_stopping,
                checkpoint
            ]
        )

        return history


    # =================================================================================================================
    # ----------------------------------------------- Private functions -----------------------------------------------
    # =================================================================================================================

    def _save_model_settings_to_csv(self, model_config: Dict[str, any]) -> None:
        """
        :return:

        Appends the model settings to a csv file.
        """

        model_config.pop('self')
        new_model_config: Dict[str, any] = {'model_ID': self._model_ID}
        new_model_config.update(model_config)

        csv_headers: List[str] = []

        for key, value in new_model_config.items():  # Gather column headers.
            csv_headers.append(key)

        with open(self._path_to_csv_logs, 'a', newline='') as file_handler:
            dict_writer = DictWriter(file_handler, fieldnames=csv_headers)

            # Write column headers if the file is new.
            if os.stat(self._path_to_csv_logs).st_size == 0:
                dict_writer.writeheader()

            dict_writer.writerow(new_model_config)


    # =================================================================================================================
    # ----------------------------------------------- Getter/Setter functions -----------------------------------------
    # =================================================================================================================

    @property
    def current_model(self) -> Union[Model, None]:
        """
        :return: A tensorflow.keras.models.Model or None if a model hasn't been set.

        Returns the last model that was constructed by the model manager. None will be returned if no model has
        been built yet.
        """

        return self._current_model


    @current_model.setter
    def current_model(self, model: Model) -> None:
        """
        :param model: a compiled tensorflow.keras.models.Model ready for fitting.
        :return:

        Updates the model stored by the manager with a model that is passed in.
        """

        # Type check.
        if isinstance(model, Model) is False:
            raise ValueError('The current model must be a compiled tensorflow.keras.models.Model')

        self._current_model = model
