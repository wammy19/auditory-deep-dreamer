import IPython.display as display
from IPython.core.display import display as core_display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from tensorflow import Tensor
from typing import Union
from .constants import SAMPLE_RATE


def display_mel_spectrogram(mel_spectrogram: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """
    :param mel_spectrogram: np.ndarray that is returned from librosa.feature.melspectrogram
    :param sample_rate: sample rate.
    :return: None

    Displays a mel spectrogram.
    """

    plt.clf()  # Clear plot.

    fig, ax = plt.subplots()
    S_dB: np.ndarray = librosa.power_to_db(mel_spectrogram, ref=np.max)

    image = librosa.display.specshow(
        S_dB,
        x_axis='time',
        y_axis='mel',
        sr=sample_rate,
        fmax=8000,
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
