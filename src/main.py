from ai_tools.model_builders import build_conv2d_model
import settings as sett
from ai_tools import ModelManager
from typing import List
import numpy as np
import os
from os.path import join
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from random import shuffle
from tensorflow.keras.models import Model


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
    training_batch_size: int = 32

    X, y = load_data(sett.dataset_path)  # type: np.ndarray, np.ndarray
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)  # type: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.2)  # type: np.ndarray, np.ndarray, np.ndarray, np.ndarray

    model_params = dict(
        num_conv_block=9,
        num_filters=128,
        num_dense_layers=2,
        dense_layer_units=64,
        conv_dropout_amount=0.1,
        num_classes=15,
    )

    model_manager = ModelManager(
        model_builder_func=build_conv2d_model,
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

    model: Model = model_manager.build_model(**model_params)
    model_manager.train_model(model, early_stopping_patience=10)
    model_manager.current_model.evaluate(X_test, y_test)

    # model: Model = build_conv2d_model(**model_params)
    #
    # model.fit(
    #     X,
    #     y,
    #     validation_split=0.2,
    #     epochs=100,
    #     verbose=True,
    #     batch_size=64
    # )

    # model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
