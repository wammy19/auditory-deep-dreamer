from ai_tools import DataGenerator
from model_utils import build_simple_cnn
from tensorflow.keras.models import Sequential


def train():
    """
    :return:
    """

    train_data_generator, val_data_generator = DataGenerator.from_path_to_audio('../data-sets/small_nsynth/train')

    model: Sequential = build_simple_cnn()

    history = model.fit(
        train_data_generator,
        validation_data=val_data_generator,
        epochs=3
    )
