import os

import tensorflow as tf
from tensorflow.keras.models import Model

import settings as sett
from ai_tools import DataGenerator, ModelManager
from ai_tools.model_builders import bayesian_optimization_test_model, build_conv2d_example


def main() -> None:
    the_meaning_of_life: int = 42  # Random seed.
    training_batch_size: int = 32
    num_samples_per_instrument: int = 2_000

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
    model_builder_params = dict(
        max_units=5_000,
        neuron_pct=0.1,
        neuron_shrink=0.5
    )

    # model_builder_params = dict(
    #
    # )

    model: Model = model_manager.build_model(**model_builder_params)
    model_manager.train_model(model)

    # # Parameters from model build function to optimize.
    # p_bounds: dict = {
    #     'num_conv_block': (3, 10),
    #     'dense_dropout_amount': (0, 0.4),
    #     'conv_dropout_amount': (0, 0.3),
    #     'regularization_amount': (0, 0.01),
    # }
    #
    # # Reduces the bounds declared above during optimization to quickly diverge towards optimal points.
    # # Resources: https://github.com/fmfn/BayesianOptimization/blob/master/examples/domain_reduction.ipynb
    # bounds_transformer = SequentialDomainReductionTransformer()
    #
    # # Create optimizer object.
    # optimizer = BayesianOptimization(
    #     f=model_manager.search_for_best_model,
    #     pbounds=p_bounds,
    #     random_state=the_meaning_of_life,
    #     bounds_transformer=bounds_transformer,
    # )
    #
    # optimizer.maximize(
    #     init_points=2,
    #     n_iter=10,
    # )


if __name__ == '__main__':
    # Only log errors.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    main()
