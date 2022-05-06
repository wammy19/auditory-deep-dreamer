import os

import tensorflow as tf
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.model_builders import *


def main() -> None:
    the_meaning_of_life: int = 42
    training_batch_size: int = 64
    num_samples_per_instrument: int = 5_000

    # Create data generators.
    train_data, val_data, test_data = DataGenerator.create_train_val_test_data_generators(
        sett.dataset_path,
        num_samples_per_instrument,
        sett.logs_path,
        training_batch_size
    )

    # Set up model manager.
    model_manager = ModelManager(
        build_conv2d_model,
        train_data,
        val_data,
        test_data,
        sett.logs_path,
        sett.model_checkpoint_path,
        training_batch_size
    )

    # Parameters from model build function to optimize.
    p_bounds = dict(
        num_conv_block=(0.0, 7),
        num_filters=(0.0, 128),
        conv_dropout_amount=(0.0, 0.499),
        num_dense_layers=(0.0, 2.0),
        dense_dropout_amount=(0.0, 0.499),
        learning_rate=(0.0, 0.1),
        kernel_regularization=(0.0, 0.001),
        activity_regularization=(0.0, 0.001),
    )

    # Create optimizer object.
    optimizer = BayesianOptimization(
        f=model_manager.build_train_and_evaluate_model,
        pbounds=p_bounds,
        random_state=the_meaning_of_life,
        bounds_transformer=SequentialDomainReductionTransformer(),
    )

    optimizer.maximize(
        init_points=15,
        n_iter=50,
    )


if __name__ == '__main__':
    # Only log errors.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    main()
