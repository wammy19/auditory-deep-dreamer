from random import choice, randint

from pandas import DataFrame

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test


def main():
    """
    :return:
    """

    # Create dataset dataframe and split it into train, validation, and test.
    df: DataFrame = create_data_frame_from_path(
        sett.dataset_path,
        number_of_samples_for_each_class=5_000
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Create Generators.
    batch_size: int = 32
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    val_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)

    # Train models using random search.
    for _ in range(100):
        model_manager = ModelManager(
            path_to_csv_logs=sett.model_config_csv_log_path,
            model_checkpoint_dir=sett.model_checkpoint_path,
            aim_logs_dir=sett.aim_logs_path
        )

        model_manager.build_model(
            num_conv_block=randint(1, 9),
            num_filters=choice([8, 16, 32, 64, 128]),
            dense_layer_size=choice([8, 16, 32, 64, 128]),
            num_dense_layers=randint(0, 5),
            use_separable_conv_layer=choice([False, True]),
            use_regularization=choice([False, True]),
            use_dropout_dense_layers=choice([False, True]),
            use_dropout_conv_blocks=choice([False, True])
        )

        model_manager.train_and_optimize_model(train_data_generator, val_data_generator)


if __name__ == '__main__':
    main()
