"""
Class used to amplify features learnt in a CNN's filter onto and input audio signal using gradient ascent.
"""

import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from librosa.feature.inverse import mel_to_audio
from maad.util import crossfade_list
from tensorflow import GradientTape, reduce_mean
from tensorflow.keras.models import Model

import utils.constants as consts
from utils.audio_tools import convert_signal_into_mel_spectrogram_segments


class ModelFilterAmplifier:
    """
    Amplifies features stored in individual filters of a CNN.
    """


    # =================================================================================================================
    # ---------------------------------------------- Dunder Methods  --------------------------------------------------
    # =================================================================================================================

    def __init__(
            self,
            model: Model,
            padding_amount: int,
            layer: int = 0
    ):

        self._model: Model = model
        self._model_layers: List[Model] = self._get_model_layers()
        self._conv_block_indicis: List[int] = self._get_conv_layer_indices()
        self._current_layer: Model = self._model.get_layer(index=self._conv_block_indicis[layer])
        self._feature_extractor = Model(inputs=self._model.inputs, outputs=self._current_layer.output)
        self._padding_amount: int = padding_amount

        self.max_num_neurons: int = len(self._model.layers) - 1


    def __call__(
            self,
            input_signal: np.ndarray,
            filter_index: int,
            learning_rate: float = 10.0,
            iterations: int = 1,
            filter_amount: float = 0.0,
            cross_fade_time: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param input_signal:
        :param filter_index:
        :param learning_rate:
        :param iterations:
        :param filter_amount:
        :param cross_fade_time:
        :return:
        """

        mel_spec_segments: List[np.ndarray] = convert_signal_into_mel_spectrogram_segments(input_signal)
        mel_specs: List[np.ndarray] = []
        processed_signal: List[np.ndarray] = []

        for mel in mel_spec_segments:
            neuron_feature: np.ndarray = self._visualize_filter(
                filter_index,
                mel,
                learning_rate=learning_rate,
                iterations=iterations,
                filter_amount=filter_amount
            )

            neuron_feature = neuron_feature.reshape(consts.NUM_MELS, -1)

            # convert mel spectrogram into audio.
            signal: np.ndarray = mel_to_audio(
                neuron_feature,
                sr=consts.SAMPLE_RATE,
                n_iter=consts.MEL_TO_AUDIO_N_ITERATIONS
            )

            # Collect audio and spectrogram segments for later joining.
            processed_signal.append(signal)
            mel_specs.append(neuron_feature)

        final_signal: np.ndarray = crossfade_list(processed_signal, 1, cross_fade_time)
        mel_specs_concat: np.ndarray = np.concatenate(np.stack(mel_specs))

        return final_signal, mel_specs_concat


    # =================================================================================================================
    # ----------------------------------------------- Public functions ------------------------------------------------
    # =================================================================================================================

    def set_current_layer(self, layer_index: int):
        """
        :param layer_index:
        :return:
        """

        try:
            self._current_layer = self._model.get_layer(
                name=self._model_layers[self._conv_block_indicis[layer_index]].name
            )

        except IndexError:
            print(
                f'No Conv layer exists at index {layer_index}. Max index number is: '
                f'{len(self._conv_block_indicis) - 1}'
            )

        self._feature_extractor = Model(inputs=self._model.inputs, outputs=self._current_layer.output)


    # =================================================================================================================
    # ----------------------------------------------- Private functions -----------------------------------------------
    # =================================================================================================================

    def _visualize_filter(
            self,
            filter_index: int,
            signal: np.ndarray,
            learning_rate: float = 10.0,
            iterations: int = 30,
            filter_amount: float = 0.0
    ) -> np.ndarray:
        """
        :param filter_index:
        :param signal:
        :param learning_rate:
        :param iterations:
        :param filter_amount:
        :return:
        """

        for _ in range(iterations):
            signal: np.ndarray = self._gradient_ascent_step(
                signal,
                filter_index,
                learning_rate
            )

        # Decode the resulting input image.
        signal: np.ndarray = self._post_process_signal(signal[0].numpy(), filter_amount)

        return signal


    def _get_model_layers(self) -> List[Model]:
        """
        :return:
        """

        layer_output: List[KerasTensor] = [layer.output for layer in self._model.layers[:len(self._model.layers) - 1]]
        activation_model = Model(inputs=self._model.input, outputs=layer_output)

        return activation_model.layers


    def _get_conv_layer_indices(self) -> List[int]:
        """
        :return:
        """

        conv_block_indices: List[int] = []

        for layer_i, layer in enumerate(self._model_layers):
            if 'conv' in layer.name:
                conv_block_indices.append(layer_i)

        return conv_block_indices


    def _compute_loss(
            self,
            input_signal: np.ndarray,
            filter_index: int
    ) -> float:
        """
        :param input_signal:
        :param filter_index:
        :return:
        """

        activation: List[np.ndarray] = self._feature_extractor(input_signal)

        try:
            filter_activation = activation[
                                :,
                                self._padding_amount: -self._padding_amount,
                                self._padding_amount: -self._padding_amount,
                                filter_index
                                ]

        except ValueError or IndexError:
            print(
                f'Filter index is set out of bounds. Max filter index possible is: '
                f'{self._current_layer.output.shape[-1] - 1}'
            )

            sys.exit(1)

        return reduce_mean(filter_activation)


    @tf.function
    def _gradient_ascent_step(
            self,
            signal: np.ndarray,
            filter_index: int,
            learning_rate: float
    ) -> np.ndarray:
        """
        :param signal:
        :param filter_index:
        :param learning_rate:
        :return:
        """

        with GradientTape() as tape:
            tape.watch(signal)
            loss: float = self._compute_loss(signal, filter_index)

        # Compute gradients.
        grads = tape.gradient(loss, signal)

        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        signal += learning_rate * grads

        return signal


    @staticmethod
    def _post_process_signal(signal: np.ndarray, filter_amount: float = 0.0) -> np.ndarray:
        """
        :param signal:
        :param filter_amount:
        :return:
        """

        signal -= filter_amount
        signal = np.clip(signal, 0, 1)

        return signal
