from os.path import join
from typing import List, Optional

import IPython.display as display
import numpy as np
import soundfile as sf
from IPython.core.display import display as core_display
from librosa import load, tone
from librosa.effects import time_stretch
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio
from librosa.util import fix_length

import utils.constants as consts
from utils.helpers import unix_url_end_filename_pattern


def segment_signal(
        audio_signal: np.ndarray,
        sample_rate: int = consts.SAMPLE_RATE,
        window_leap_fraction: int = 2
) -> List[np.ndarray]:
    """
    :param audio_signal: np.ndarray of audio returned from librosa.load()
    :param sample_rate: Sample rate of audio that has been loaded.
    :param window_leap_fraction: Amount the window should move in relation to the sample rate. Default at 2, therefor
    window will move at half the sample rate.
    :return: List[np.ndarray] - List of 1 second slices of audio.

    Slices up an audio signal into segments that are all 1 second long. Returns a List with each 1 second segment.
    Padding is added to end of last segment if necessary.
    """

    start_index: int = 0
    all_audio_segments: List[np.ndarray] = []

    # Pad audio with 0's if it's less than a second and return signal.
    if audio_signal.size < consts.SAMPLE_RATE:
        sample: np.ndarray = fix_length(audio_signal, size=consts.SAMPLE_RATE)
        all_audio_segments.append(sample)

        return all_audio_segments

    # Loop through signal and chop it up into 1 second segments.
    while start_index < audio_signal.size:
        one_second_sample: np.ndarray = audio_signal[start_index: start_index + consts.SAMPLE_RATE]
        one_second_sample = fix_length(one_second_sample, size=consts.SAMPLE_RATE)  # Ensure signal is a second-long.

        all_audio_segments.append(one_second_sample)
        start_index += (sample_rate // window_leap_fraction)

    return all_audio_segments


def convert_signal_into_mel_spectrogram_segments(audio_signal: np.ndarray) -> List[np.ndarray]:
    """
    :param audio_signal:
    :return:
    """

    # Segment signal into 1 second samples
    signal_segments: List[np.ndarray] = segment_signal(audio_signal, window_leap_fraction=1)
    mel_spec_segments: List[np.ndarray] = []

    # Loop over all segments and encode them into mel spectrograms.
    for sample in signal_segments:
        encoded_segment: np.ndarray = melspectrogram(
            y=sample,
            sr=consts.SAMPLE_RATE,
            n_fft=consts.NUM_FFT,
            hop_length=consts.MEL_HOP_LEN,
            n_mels=consts.NUM_MELS,
            win_length=consts.MEL_WINDOW_LEN
        )

        # Reshape mel spectrograms
        # TODO: Investigate if wrapping this numpy array in another array produces the correct
        #  shape for model to process. resulting shape: (1, 300, 44, 1).
        encoded_segment = np.array([encoded_segment.reshape(consts.X_SHAPE)])
        mel_spec_segments.append(encoded_segment)

    return mel_spec_segments


def convert_signal_into_mel_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    :param signal:
    :return:
    """

    return melspectrogram(
        y=signal,
        sr=consts.SAMPLE_RATE,
        n_fft=consts.NUM_FFT,
        hop_length=consts.MEL_HOP_LEN,
        n_mels=consts.NUM_MELS,
        win_length=consts.MEL_WINDOW_LEN
    )


def create_audio_player(
        signal: np.ndarray,
        sample_rate: int = consts.SAMPLE_RATE,
        normalize: bool = False,
        mel_n_iterations: int = consts.MEL_TO_AUDIO_N_ITERATIONS
) -> None:
    """
    :param signal: A 1D raw audio signal, or a mel spectrogram.
    :param sample_rate: Sample rate of signal.
    :param normalize: Normalize audio. Default is False.
    :param mel_n_iterations:
    :return:

    Displays an audio player for playback.
    """

    if len(signal.shape) > 2:
        try:
            signal = signal.reshape(consts.NUM_MELS, -1)

        except ValueError as err:
            print('Signal must be either 1, 2 or 3 dimensional to be converted back into audio.')
            print(err)

    if 1 < len(signal.shape) < 3:  # If signal is a Mel spectrogram.
        signal: np.ndarray = mel_to_audio(
            signal,
            sr=consts.SAMPLE_RATE,
            n_iter=mel_n_iterations
        )

    core_display(display.Audio(signal, rate=sample_rate, normalize=normalize))


def time_stretch_signal(path_to_sample: str, sample: str, instrument, path_for_writing: str):
    """
    :param path_to_sample:
    :param sample:
    :param instrument:
    :param path_for_writing:
    :return:

    Creates a time stretches version of a sample.
    """

    loaded_sample: np.ndarray = load(join(path_to_sample, sample), sr=consts.SAMPLE_RATE, mono=True)[0]
    stretched_sample: np.ndarray = time_stretch(loaded_sample, rate=0.5)
    file_name: str = f'{unix_url_end_filename_pattern.findall(sample)[0]}_stretched.wav'

    sf.write(join(path_for_writing, instrument, file_name), stretched_sample, consts.SAMPLE_RATE)


def load_and_convert_audio_into_mel_spectrogram(
        path_to_data: str,
        reshape: bool = False,
        duration: Optional[float] = None
) -> np.ndarray:
    """
    :param path_to_data - Path to wav file.
    :param reshape - Reshape spectrogram for be digested by model.
    :param duration:
    :return:

    Load an audio file and encode it into a mel spectrogram.
    """

    sample: np.ndarray = load(path_to_data, mono=True, duration=duration)[0]
    mel_spectrogram: np.ndarray = melspectrogram(
        y=sample,
        sr=consts.SAMPLE_RATE,
        n_fft=consts.NUM_FFT,
        hop_length=consts.MEL_HOP_LEN,
        n_mels=consts.NUM_MELS,
        win_length=consts.MEL_WINDOW_LEN
    )

    if reshape:
        mel_spectrogram = mel_spectrogram.reshape((1, 300, 44, 1))

    return mel_spectrogram


def generate_sine_wave_mel_spectrogram(
        freq: int = 440,
        duration: float = 1.0,
        sample_rate: int = consts.SAMPLE_RATE
) -> np.ndarray:
    """
    :param freq: Frequency in Hz. 440hz == A
    :param duration: Time in seconds.
    :param sample_rate:
    :return:

    Generates a sine wave and encodes it into a mel spectrogram.
    """

    sine_wave: np.ndarray = tone(freq, duration=duration, sr=sample_rate)

    encoded_sine_wave: np.ndarray = melspectrogram(
        y=sine_wave,
        sr=consts.SAMPLE_RATE,
        n_fft=consts.NUM_FFT,
        hop_length=consts.MEL_HOP_LEN,
        n_mels=consts.NUM_MELS,
        win_length=consts.MEL_WINDOW_LEN
    )

    return np.array([encoded_sine_wave.reshape(consts.X_SHAPE)])
