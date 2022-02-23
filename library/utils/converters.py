from librosa import power_to_db
from librosa.feature import melspectrogram
import PIL.Image
import numpy as np
from typing import Optional
import utils.constants as consts


def open_image_as_np_array(path_to_img: str, max_dim: Optional[int] = None) -> np.ndarray:
    """
    :param: path_to_img: Path to an image.
    :param: max_dim: Maximum dimensions of image.
    :return:

    Loads an image and converts it into a numpy array. If a value for max_dim is passed, the image
    will be resized.
    """

    _image: PIL.Image = PIL.Image.open(path_to_img)

    if max_dim is not None:
        _image.thumbnail((max_dim, max_dim))

    return np.array(_image)


def convert_wav_as_log_mel(
        sample: np.ndarray,
        sample_rate: int = consts.SAMPLE_RATE,
        num_FFT: int = consts.NUM_FFT,
        mel_hop_len: int = consts.MEL_HOP_LEN,
        num_mels: int = consts.NUM_MELS
) -> np.ndarray:
    """
    :param: sample: numpy array that is returned from librosa.load()[0]
    :param: sample_rate:
    :param: num_FFT:
    :param: mel_hop_len:
    :param: num_mels:
    :return:

    Converts a loaded wav file into a log mel-spectrogram.
    """

    encoded_sample: np.ndarray = melspectrogram(
        y=sample,
        sr=sample_rate,
        n_fft=num_FFT,
        hop_length=mel_hop_len,
        n_mels=num_mels
    )

    return power_to_db(encoded_sample)
