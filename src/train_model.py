from librosa import load
from model_utils import build_simple_cnn
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
from typing import List, Tuple
import utils.constants as consts


def load_data(path_to_data: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param path_to_data: str
    :return: Tuple[]
    """

    data: List[np.ndarray] = []
    labels: List[int] = []

    for i, category in enumerate(os.listdir(path_to_data)):
        path_to_category = os.path.join(path_to_data, category)

        for audio in os.listdir(path_to_category):
            audio, _ = load(os.path.join(path_to_category, audio), sr=consts.SAMPLE_RATE)
            data.append(audio)
            labels.append(i)

    return np.asarray(data), np.asarray(labels)


def train():
    """
    :return:
    """

    model: Sequential = build_simple_cnn()

    print('Loading data...')
    X_train, y_train = load_data('../data-sets/small_nsynth/train')
    # X_val, y_val = load_data('../data-sets/small_nsynth/validation')

    print('Training model.')
    history: History = model.fit(
        X_train,
        y_train,
        batch_size=512,
        validation_split=0.1,
        # validation_data=(X_val, y_val),
        epochs=5
    )

    print(history)
