import sys
import numpy as np
from keras.engine.keras_tensor import KerasTensor
from tensorflow.keras.models import Model
from typing import List, Tuple
from tensorflow import GradientTape, reduce_mean
import tensorflow as tf
import utils.constants as consts
from librosa.feature.inverse import mel_to_audio
from utils.audio_tools import convert_signal_into_mel_spectrogram_segments


class ModelFilterAmplifier:
    """
    Amplifies features stored in individual filters of a CNN.
    """


    def __int__(
            self,
            model: Model,
    ):

        self._model: Model = model
        self._model_layers: List[Model] = self._get_model_layers(self._model)
        self._conv_block_indicis: List[int] = self._get_conv_layer_indicis(self._model_layers)

        self._current_layer = self._model.get_layer(name=self._model_layers[self._conv_block_indicis[0]].name)
        self._feature_extractor = Model(input_shape=self._model.inputs, outputs=self._current_layer.output)


    def __call__(
            self,
            signal: np.ndarray,
            filter_index: int,
            learning_rate: float = 10.0,
            iterations: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param filter_index:
        :param learning_rate:
        :param iterations:
        :param signal:
        :return:
        """

        # Segment audio signal and convert segments into mel spectrograms
        mel_spec_segments: List[np.ndarray] = convert_signal_into_mel_spectrogram_segments(signal)
        mel_specs: List[np.ndarray] = []
        processed_signal: List[np.ndarray] = []

        for mel in mel_spec_segments:

            # Perform gradient ascent.
            for _ in range(iterations):
                mel: np.ndarray = self._gradient_ascent_step(
                    self._feature_extractor,
                    mel,
                    filter_index,
                    learning_rate
                )

            # Decode the resulting input image.
            mel = self._post_process_signal(signal[0].numpy())
            neuron_feature = mel.reshape(consts.NUM_MELS, -1)

            # convert mel spectrogram into audio.
            signal: np.ndarray = mel_to_audio(
                neuron_feature,
                sr=consts.SAMPLE_RATE,
                n_iter=consts.MEL_TO_AUDIO_N_ITERATIONS
            )

            # Collect audio and spectrogram segments for later joining.
            processed_signal.append(signal)
            mel_specs.append(neuron_feature)

        # Put together all audio and mel spectrogram signals.
        final_signal = np.concatenate(np.stack(processed_signal))
        mel_specs_concate: np.ndarray = np.concatenate(np.stack(mel_specs))

        return final_signal, mel_specs_concate


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

        self._feature_extractor = Model(input_shape=self._model.inputs, outputs=self._current_layer.output)


    @staticmethod
    def _get_model_layers(model: Model) -> List[Model]:
        """
        :param model:
        :return:
        """

        layer_output: List[KerasTensor] = [_layer.output for _layer in model.layers[:len(model.layers) - 1]]
        activation_model = Model(inputs=model.input, outputs=layer_output)

        return activation_model.layers


    @staticmethod
    def _get_conv_layer_indicis(model_layers: List[Model]) -> List[int]:
        """
        :param model_layers:
        :return:
        """

        conv_block_indicis: List[int] = []

        for i, layer in enumerate(model_layers):
            if 'conv' in layer.name:
                conv_block_indicis.append(i)

        return conv_block_indicis


    def _compute_loss(
            self,
            _feature_extractor: Model,
            _input_signal: np.ndarray,
            _filter_index: int
    ) -> float:
        """
        :param _feature_extractor:
        :param _input_signal:
        :param _filter_index:
        :return:
        """

        activation: List[np.ndarray] = _feature_extractor(_input_signal)

        try:
            filter_activation = activation[:, :, :, _filter_index]

        except IndexError:
            print(
                f'Filter index is set out of bounds. Max filter index possible is: '
                f'{self._current_layer.output.shape[-1] - 1}'
            )

            sys.exit(2)

        return reduce_mean(filter_activation)


    @tf.function
    def _gradient_ascent_step(
            self,
            feature_extractor: Model,
            signal: np.ndarray,
            filter_index: int,
            learning_rate: float
    ) -> np.ndarray:
        """
        :param feature_extractor:
        :param signal:
        :param filter_index:
        :param learning_rate:
        :return:
        """

        with GradientTape() as tape:
            tape.watch(signal)
            _loss = self._compute_loss(feature_extractor, signal, filter_index)

        # Compute gradients.
        grads = tape.gradient(_loss, signal)

        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        signal += learning_rate * grads

        return signal


    @staticmethod
    def _post_process_signal(signal: np.ndarray) -> np.ndarray:
        """
        :param signal:
        :return:
        """

        # Normalize array: center on 0., ensure variance is 0.15.
        signal -= signal.mean()
        signal /= signal.std() + 1e-5
        signal *= 0.15

        # Clip to [0, 1].
        signal = np.clip(signal, 0, 1)

        return signal
