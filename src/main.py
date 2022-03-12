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

    # Train models using random search.
    for _ in range(100):
        model_manager = ModelManager(
            path_to_csv_logs=sett.model_config_csv_log_path,
            model_checkpoint_dir=sett.model_checkpoint_path,
            aim_logs_dir=sett.aim_logs_path,
            history_log_dir=sett.model_histories
        )

        # Bounded region of parameter space
        p_bounds: dict = {'num_conv_block': (2, 12), 'num_filters': (32, 128)}

        # optimizer = BayesianOptimization(
        #     f=model_manager.build_model,
        #     pbounds=p_bounds,
        #     random_state=1,
        # )

        model_manager.build_model(
            num_conv_block=randint(1, 15),
            num_filters=choice([64, 128]),
            dense_layer_size=choice([64, 128]),
            num_dense_layers=randint(0, 5),
            use_separable_conv_layer=False,
            use_regularization=choice([False, True]),
            use_dropout_dense_layers=choice([False, True]),
            use_dropout_conv_blocks=choice([False, True])
        )

        accuracy: float = model_manager.train_and_optimize_model(
            train_data_generator,
            val_data_generator,
            test_data_generator,
            epochs=100
        )


if __name__ == '__main__':
    main()
