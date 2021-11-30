import IPython.display as display
from IPython.core.display import display as core_display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from tensorflow import Tensor
from typing import Union
from .constants import SAMPLE_RATE


def display_mel_spectrogram(mel_spec: np.ndarray, is_log_mel: bool, sample_rate: int = SAMPLE_RATE) -> None:
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

    image = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        ax=ax
    )

    fig.colorbar(image, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def display_image(_image: Union[np.array, Tensor]) -> None:
    """
    :param _image: Image that's been converted into a numpy array.
    :return:

    Displays an image that's been converted into a numpy array.
    """

    display.display(PIL.Image.fromarray(np.array(_image)))


def display_audio_player(audio_file: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """
    :return:

    Displays an audio player.
    """

    core_display(display.Audio(audio_file, rate=sample_rate))