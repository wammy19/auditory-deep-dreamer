from librosa.util import fix_length
import numpy as np
from typing import List
import utils.constants as consts


def segment_signal(
        audio_signal: np.ndarray,
        sample_rate: int = consts.SAMPLE_RATE,
        window_leap_fraction: int = 2
) -> List[np.ndarray]:
    """
    :param: audio_signal: np.ndarray of audio returned from librosa.load()
    :param: sample_rate: Sample rate of audio that has been loaded.
    :param: window_leap_fraction: Amount the window should move in relation to the sample rate. Default at 2, therefor
    window will move at half the sample rate.
    :return: List[np.ndarray] - List of 1 second slices of audio.

    Slices up an audio signal into segments that are all 1 second long. Returns a List with each 1 second segment.
    Padding is added to end of last segment if necessary.
    """

    start_index = 0
    all_audio_segments: List[np.ndarray] = []

    # Pad audio with 0's if it's less than a second and return signal.
    if audio_signal.size < consts.SAMPLE_RATE:
        sample = fix_length(audio_signal, consts.SAMPLE_RATE)
        all_audio_segments.append(sample)

        return all_audio_segments

    # Loop through signal and chop it up into 1 second segments.
    while start_index < audio_signal.size:
        one_second_sample: np.ndarray = audio_signal[start_index: start_index + consts.SAMPLE_RATE]
        one_second_sample = fix_length(one_second_sample, consts.SAMPLE_RATE)  # Ensure signal is a second-long.

        all_audio_segments.append(one_second_sample)
        start_index += (sample_rate // window_leap_fraction)

    return all_audio_segments
