from os.path import join
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
        number_of_samples_for_each_class=2_000
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Store the data generator data frame for recreating the data generator if needed.
    df_train.to_csv(join(sett.logs_path, 'train_data.csv'))
    df_val.to_csv(join(sett.logs_path, 'val_data.csv'))
    df_test.to_csv(join(sett.logs_path, 'test_data.csv'))

    # Create Generators.
    batch_size: int = 32
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    val_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)
    test_data_generator: DataGenerator = DataGenerator(df_test, batch_size=batch_size)

    model_manager = ModelManager(
        path_to_logs=sett.logs_path,
        model_checkpoint_dir=sett.model_checkpoint_path,
    )

    # Train models using random search.
    for _ in range(100):
        model_manager.search_for_best_model(
            train_data_generator,
            val_data_generator,
            test_data_generator,
            num_conv_block=randint(1, 9),
            num_filters=choice([8, 16, 32, 64, 128]),
            dense_layer_size=choice([8, 16, 32, 64, 128]),
            num_dense_layers=randint(0, 5),
            use_separable_conv_layer=choice([False, True]),
            use_regularization=choice([False, True]),
            use_dropout_dense_layers=choice([False, True]),
            use_dropout_conv_blocks=choice([False, True])
        )


if __name__ == '__main__':
    main()
