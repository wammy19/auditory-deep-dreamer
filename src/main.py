from os.path import join
import os
from random import choice, randint

from bayes_opt import BayesianOptimization
from pandas import DataFrame
import tensorflow as tf

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test

from ai_tools.model_builders import dynamic_conv2d_model


def main():
    """
    :return:
    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only log errors.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Create dataset dataframe and split it into train, validation, and test.
    df: DataFrame = create_data_frame_from_path(
        sett.dataset_path,
        number_of_samples_for_each_class=10_000
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Store the data generator data frame for recreating the data generator if needed.
    df_train.to_csv(join(sett.logs_path, 'train_data.csv'))
    df_val.to_csv(join(sett.logs_path, 'val_data.csv'))
    df_test.to_csv(join(sett.logs_path, 'test_data.csv'))

    # Create Generators.
    batch_size: int = 32
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    validation_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)
    test_data_generator: DataGenerator = DataGenerator(df_test, batch_size=batch_size)

    model_manager = ModelManager(
        train_data_generator,
        validation_data_generator,
        test_data_generator,
        path_to_logs=sett.logs_path,
        model_checkpoint_dir=sett.model_checkpoint_path,
    )

    p_bounds: dict = {
        'dropout_amount': (0, 0.6),
        'learning_rate': (0, 0.01)
    }

    optimizer = BayesianOptimization(
        f=model_manager.search_for_best_model,
        pbounds=p_bounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=25,
    )


if __name__ == '__main__':
    main()
