from ai_tools.model_builders import build_conv2d_model
from ai_tools import DataGenerator
import settings as sett
from ai_tools import ModelManager


def main() -> None:
    dataset_path: str = '/home/andrea/dev/uni/datasets/serialized_philharmonia_dataset'

    training_batch_size: int = 62
    num_samples_per_instrument: int = 10_000

    # Create data generators.
    train_data, val_data, test_data = DataGenerator.create_train_val_test_data_generators(
        dataset_path,
        num_samples_per_instrument,
        training_batch_size
    )

    model_manager = ModelManager(
        build_conv2d_model,
        train_data,
        val_data,
        test_data,
        sett.logs_path,
        sett.model_checkpoint_path,
        training_batch_size
    )

    model_params = dict(
        num_conv_block=9,
        num_filters=128,
        num_dense_layers=2,
        dense_layer_units=64,
        conv_dropout_amount=0.1,
        num_classes=15,
    )

    model = model_manager.build_model(**model_params)
    model_manager.train_model(model)

    model.evaluate(test_data)


if __name__ == '__main__':
    main()
