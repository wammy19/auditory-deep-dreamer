import os
from dataclasses import dataclass
from os.path import join
from random import shuffle
from typing import List

import numpy as np
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from tensorflow.keras.utils import to_categorical

import settings as sett
from ai_tools import ModelManager
from ai_tools.model_builders import vgg_like_model


@dataclass
class Data:
    data: np.ndarray
    label: np.ndarray


def load_data(path_to_data_set: str):
    """
    :param path_to_data_set:
    :return:
    """

    ontology: List[str] = os.listdir(path_to_data_set)
    ontology_len: int = len(ontology)

    data: List[Data] = []

    for i, instrument in enumerate(ontology):
        path_to_instrument: str = join(path_to_data_set, instrument)
        samples: List[str] = os.listdir(path_to_instrument)

        for sample in samples:
            path_to_sample: str = join(path_to_instrument, sample)
            mel_spec: np.ndarray = np.load(path_to_sample)
            label: np.ndarray = to_categorical(i, ontology_len)

            data.append(Data(mel_spec, label))

    shuffle(data)

    X_data: np.ndarray = np.stack([x.data for x in data])
    y_data: np.ndarray = np.stack([y.label for y in data])

    return X_data, y_data


def main() -> None:
    the_meaning_of_life: int = 42
    training_batch_size: int = 64

    X, y = load_data(join(sett.dataset_path, 'train'))
    X_val, y_val = load_data(join(sett.dataset_path, 'validation'))
    X_test, y_test = load_data(join(sett.dataset_path, 'test'))

    model_manager = ModelManager(
        model_builder_func=vgg_like_model,
        train_data=X,
        validation_data=X_val,
        test_data=X_test,
        train_labels=y,
        validation_labels=y_val,
        test_labels=y_test,
        path_to_logs=sett.logs_path,
        model_checkpoint_dir=sett.model_checkpoint_path,
        training_batch_size=training_batch_size
    )

    p_bounds = dict(
        num_first_conv_blocks=(1, 9),
        num_second_conv_blocks=(1, 9),
        num_third_conv_blocks=(1, 9),
        num_fourth_conv_blocks=(1, 9),
        dropout_amount=(0.0, 0.499),
        learning_rate=(0.0, 0.0001),
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
    main()
