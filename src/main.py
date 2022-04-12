import os

import tensorflow as tf
from bayes_opt import SequentialDomainReductionTransformer, BayesianOptimization
from tensorflow.keras.models import Model

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.model_builders import bayesian_optimization_test_model, build_conv2d_example, vgg_like_model


def main() -> None:
    the_meaning_of_life: int = 42
    training_batch_size: int = 32
    num_samples_per_instrument: int = 10

    # Create data generators.
    train_data, val_data, test_data = DataGenerator.create_train_val_test_data_generators(
        sett.dataset_path,
        num_samples_per_instrument,
        sett.logs_path,
        training_batch_size
    )


    # Set up model manager.
    model_manager = ModelManager(
        bayesian_optimization_test_model,
        train_data,
        val_data,
        test_data,
        sett.logs_path,
        sett.model_checkpoint_path,
        training_batch_size
    )

    # bayesian_optimization_test_model params.
    # model_builder_params = dict(
    #         neuron_pct=0.5,
    #         neuron_shrink=0.9,  # Max 0.9
    #         max_units=1_000,  # Min 1_000
    #         drop_out_amount=0.2,
    #         learning_rate=0.01,
    #         kernel_regularization=0.01,
    #         activity_regularization=0.01
    # )
    #
    # model: Model = model_manager.build_model(**model_builder_params)
    # model_manager.train_model(model)

    # Parameters from model build function to optimize.
    pbounds = dict(
        neuron_pct=(0.0, 1.0),
        neuron_shrink=(0.0, 0.9),
        max_units=(100, 1_000),
        drop_out_amount=(0.0, 0.499),
        learning_rate=(0, 0.1),
        kernel_regularization=(0.0, 0.1),
        activity_regularization=(0.0, 0.1)
    )

    # Create optimizer object.
    optimizer = BayesianOptimization(
        f=model_manager.search_for_best_model,
        pbounds=pbounds,
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
