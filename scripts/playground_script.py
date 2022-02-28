from ai_tools import DataGenerator
from utils.helpers import read_yaml_config

yaml_config: dict = read_yaml_config()
PATH_TO_DATASET: str = yaml_config['path_to_dataset']


def test_data_generator():
    """
    :return:
    """

    number_of_samples_for_each_class: int = 100_000
    batch_size: int = 1_000

    # False shuffle so that we can appropriately test that the path matches the correct audio.
    data_generator = DataGenerator.from_path_to_audio(
        PATH_TO_DATASET,
        number_of_samples_for_each_class=number_of_samples_for_each_class,
        shuffle=False,
        batch_size=batch_size,
        include_pitch_labels=True
    )

    print(f'Number of tasks: {number_of_samples_for_each_class // batch_size}')

    for batch_index in range(number_of_samples_for_each_class // batch_size):
        mel_spec = data_generator[batch_index][0]


test_data_generator()
