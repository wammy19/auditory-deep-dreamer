from ai_tools import DataGenerator
from ai_tools.models import build_simple_cnn
import sys
from tensorflow.keras.models import Sequential

sys.path.append('../library/')  # Add custom library to PYTHONPATH.


def main() -> None:
    """
    :return: None
    """

    # Create train and validation generators.
    train_data_generator, val_data_generator = DataGenerator.from_path_to_audio('../data-sets/small_nsynth/train')

    # Build model.
    model: Sequential = build_simple_cnn()

    # Train model.
    history = model.fit(
        train_data_generator,
        validation_data=val_data_generator,
        epochs=3
    )


if __name__ == "__main__":
    main()
