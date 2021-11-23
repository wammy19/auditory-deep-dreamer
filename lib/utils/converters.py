import PIL.Image
import numpy as np
from typing import Optional


def open_image_as_np_array(path_to_img: str, max_dim: Optional[int] = None) -> np.array:
    """
    :param path_to_img: Path to an image.
    :param max_dim: Maximum dimensions of image.
    :return:

    Loads an image and converts it into a numpy array. If a value for max_dim is passed, the image
    will be resized.
    """

    _image = PIL.Image.open(path_to_img)

    if max_dim is not None:
        _image.thumbnail((max_dim, max_dim))

    return np.array(_image)
