from dataclasses import dataclass
from librosa import load
from librosa.util import fix_length
import numpy as np
import os
from typing import List
import utils.constants as consts


@dataclass
class Data:
    """
    Little data structure for holding data and labels.
    """

    raw_audio: np.ndarray
    label: int


def load_data(path_to_audio: str) -> List[Data]:
    """
    :param path_to_audio:
    :return:
    """

    samples: List[Data] = []  # Little data structure for holding raw audio data and it's label.

    for label, _class in enumerate(os.listdir(path_to_audio)):
        for file in os.listdir(f'{path_to_audio}/{_class}'):
            sample: np.ndarray = load(
                f'{path_to_audio}/{_class}/{file}',
                duration=1.0,
                mono=True,
                sr=consts.SAMPLE_RATE
            )[0]

            sample = fix_length(sample, consts.SAMPLE_RATE)  # Pad audio with 0's if it's less than a second.
            samples.append(Data(sample, label))

    return samples
