from os.path import join
from random import choice, randint

from aim.keras import AimCallback
from pandas import DataFrame
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Model

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test


def main():
    df: DataFrame = create_data_frame_from_path(
        sett.dataset_path,
        number_of_samples_for_each_class=5_000
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Create Generators.
    batch_size: int = 64
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    val_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)

    num_epochs: int = 100

    for i in range(100):
        # Create model.
        model_name: str = f'model_{i}'
        model_manager = ModelManager(
            model_name=model_name,
            num_conv_block=randint(1, 9),
            num_filters=choice([8, 16, 32, 64, 128]),
            dense_layer_size=choice([8, 16, 32, 64, 128]),
            num_dense_layers=randint(0, 5),
            use_separable_conv_layer=choice([False, True]),
            use_regularization=choice([False, True]),
            use_dropout_dense_layers=choice([False, True]),
            use_dropout_conv_blocks=choice([False, True]),
        )

        model_manager.save_model_settings_to_csv(sett.model_settings_path)
        model: Model = model_manager.build_model()

        # Callbacks.
        aim_callback = AimCallback(repo=sett.aim_logs_path, experiment=model_name)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=False)
        checkpoint = ModelCheckpoint(
            join(join(sett.model_checkpoint_path, model_name), 'epoch-{epoch:02d}.pb'),
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
            epochs=num_epochs,
            validation_data=val_data_generator,
            validation_steps=len(val_data_generator.get_data_frame.index) // batch_size,
            batch_size=batch_size,
            callbacks=[
                aim_callback,
                early_stopping,
                checkpoint
            ]
        )


if __name__ == '__main__':
    main()
