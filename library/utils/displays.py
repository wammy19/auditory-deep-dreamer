from typing import Union

import IPython.display as display
import PIL.Image
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import Tensor

from .constants import SAMPLE_RATE


def display_mel_spectrogram(mel_spec: np.ndarray, is_log_mel: bool = False, sample_rate: int = SAMPLE_RATE) -> None:
    """
    :param mel_spec: np.ndarray that is returned from librosa.feature.melspectrogram
    :param is_log_mel: Set to True if you've passed the mel spectrogram through librosa.power_to_db()
    :param sample_rate: sample rate.
    :return: None

    Displays a mel spectrogram.
    """

    plt.clf()  # Clear plot.

    fig, ax = plt.subplots()

    if not is_log_mel:  # Convert mel spec to log scale.
        mel_spec: np.ndarray = librosa.power_to_db(mel_spec, ref=np.max)

    if len(mel_spec.shape) > 2:
        mel_spec = mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1])

    image = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        ax=ax,
    )

    fig.colorbar(image, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def display_image(_image: Union[np.array, Tensor]) -> None:
    """
    :param _image: Image that's been converted into a numpy array.
    :return:

    Displays an image that's been converted into a numpy array.
    """

    display.display(PIL.Image.fromarray(np.array(_image)))


def display_wave_form(audio_file: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """
    :return:
    """

    librosa.display.waveshow(audio_file, sr=sample_rate)
    plt.show()
