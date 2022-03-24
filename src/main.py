import os
from os.path import join

import tensorflow as tf
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from pandas import DataFrame

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.helpers import create_data_frame_from_path, split_stratified_into_train_val_test
from ai_tools.model_builders import vgg_like_model


def main():
    """
    :return:
    """

    # Misc settings.
    num_of_samples_per_instrument: int = 155_000  # 160_000 is max.
    the_meaning_of_life: int = 42  # Random seed.
    batch_size: int = 32

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only log errors.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Create dataset dataframe and split it into train, validation, and test.
    df: DataFrame = create_data_frame_from_path(
        sett.dataset_path,
        number_of_samples_for_each_class=num_of_samples_per_instrument
    )

    df_train, df_val, df_test = split_stratified_into_train_val_test(df)  # type: DataFrame, DataFrame, DataFrame

    # Store the data generator data frame for recreating the data generator if needed.
    df_train.to_csv(join(sett.logs_path, 'train_data.csv'))
    df_val.to_csv(join(sett.logs_path, 'val_data.csv'))
    df_test.to_csv(join(sett.logs_path, 'test_data.csv'))

    # Create Generators.
    train_data_generator: DataGenerator = DataGenerator(df_train, batch_size=batch_size)
    validation_data_generator: DataGenerator = DataGenerator(df_val, batch_size=batch_size)
    test_data_generator: DataGenerator = DataGenerator(df_test, batch_size=batch_size)

    model_manager = ModelManager(
        vgg_like_model,
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

    # Reduces the bounds declared above during optimization to quickly diverge towards optimal points.
    # Resources: https://github.com/fmfn/BayesianOptimization/blob/master/examples/domain_reduction.ipynb
    bounds_transformer = SequentialDomainReductionTransformer()

    # Create optimizer object.
    optimizer = BayesianOptimization(
        f=model_manager.search_for_best_model,
        pbounds=p_bounds,
        random_state=the_meaning_of_life,
        bounds_transformer=bounds_transformer,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )


if __name__ == '__main__':
    main()
