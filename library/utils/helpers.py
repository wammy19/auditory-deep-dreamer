from dataclasses import dataclass
import numpy as np


@dataclass
class Data:
    """
    Little data structure for holding data and labels.
    """

    raw_audio: np.ndarray
    label: int
