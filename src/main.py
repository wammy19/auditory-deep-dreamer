from keras.callbacks import History
from pandas import DataFrame

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test

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

from random import randint, choice


def main():
    df: DataFrame = create_data_frame_from_path(
        sett.dataset_path,
        number_of_samples_for_each_class=100
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Create Generators.
    batch_size: int = 32
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    val_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)

    # Create model.
    model_manager = ModelManager(
        path_to_csv_logs=sett.model_config_csv_log_path,
        model_checkpoint_dir=sett.model_checkpoint_path,
        aim_logs_dir=sett.aim_logs_path
    )

    batch_size: int = 32

    # Create a  model.
    model: Model = model_manager.build_model(
        num_conv_block=randint(1, 9),
        num_filters=choice([8, 16, 32, 64, 128]),
        dense_layer_size=choice([8, 16, 32, 64, 128]),
        num_dense_layers=randint(0, 5),
        use_separable_conv_layer=choice([False, True]),
        use_regularization=choice([False, True]),
        use_dropout_dense_layers=choice([False, True]),
        use_dropout_conv_blocks=choice([False, True]),
    )

    model_name: str = f'model_test'

    # Callbacks.
    aim_callback = AimCallback(repo=sett.aim_logs_path, experiment=f'model_test')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=False)
    checkpoint = ModelCheckpoint(
        join(join(sett.model_checkpoint_path, f'model_test'), 'epoch-{epoch:02d}.pb'),
        monitor='val_accuracy',
        verbose=False,
        save_weights_only=False,
        save_best_only=True,
        mode='max',
    )

    # Train model.
    model.fit(
        train_data_generator,
        steps_per_epoch=len(train_data_generator.get_data_frame.index) // batch_size,
        epochs=200,
        validation_data=val_data_generator,
        validation_steps=len(val_data_generator.get_data_frame.index) // batch_size,
        batch_size=batch_size,
        callbacks=[
            aim_callback,
            early_stopping,
            checkpoint
        ]
    )

    # Build a model.
    # model_manager.train_and_evaluate_model(train_data_generator, val_data_generator)


if __name__ == '__main__':
    main()
