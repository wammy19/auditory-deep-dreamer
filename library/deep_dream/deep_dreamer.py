import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from typing import List, Tuple
import numpy as np
from tensorflow import TensorSpec


class DeepDreamer(tf.Module):
    """
    This code is refactored code from the Tensorflow documentation.

    The purpose of following through this Tensorflow guide is to get an understanding on how to access a models layers
    for extracting their output.

    Source: https://www.tensorflow.org/tutorials/generative/deepdream
    """


    def __init__(self, layer_names_for_amplification: List[str]):
        """
        :param layer_names_for_amplification: Name of layers you wish to amplify. Layer name example: 'mixed0'
        layers range from 0 to 10.
        """

        super().__init__()

        # Resources:
        # Keras Documentation: https://keras.io/api/applications/inceptionv3/
        # Incite into the models' architecture: https://iq.opengenus.org/inception-v3-model-architecture/
        base_model: Model = InceptionV3(include_top=False, weights='imagenet')
        layers: List[str] = [base_model.get_layer(name).output for name in layer_names_for_amplification]

        # Create the feature extraction model.
        self.model = Model(inputs=base_model.input, outputs=layers)


    @tf.function(
        input_signature=(
                TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, image: np.ndarray, tile_size: int = 512) -> np.ndarray:
        """
        :param image: A loaded image that's been converted to a tf.Tensor.
        :param tile_size:
        :return:

        "Adding the gradients to the image enhances the patterns seen by the network. At each step,
        you will have created an image that increasingly excites the activations of certain layers in the network."
        """

        shift, img_rolled = self._random_roll(image, tile_size)  # type: np.ndarray, np.ndarray

        # Initialize the image gradients to zero.
        gradients: np.ndarray = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs: np.ndarray = tf.range(0, img_rolled.shape[0], tile_size)[:-1]

        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])

        ys: np.ndarray = tf.range(0, img_rolled.shape[1], tile_size)[:-1]

        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:  # Calculate the gradients for this tile.
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss: np.ndarray = self._calculate_loss(img_tile)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


    def dreamify(
            self,
            img: np.ndarray,
            steps_per_octave: int = 10,
            step_size: float = 0.01,
            octaves: range = range(-2, 3),
            octave_scale: float = 1.3,
    ) -> np.ndarray:
        """
        :param img:
        :param steps_per_octave:
        :param step_size:
        :param octaves:
        :param octave_scale:
        :return:

        Will create a deep dream image returned as a tf.Tensor.
        """

        base_shape: np.ndarray = tf.shape(img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)

        for octave in octaves:
            # Scale the image based on the octave
            new_size: float = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

            for step in range(steps_per_octave):
                gradients = self(img)
                img += gradients * step_size
                img = tf.clip_by_value(img, -1, 1)

        result: np.ndarray = self._normalize_image(img)

        return result


    def get_layer_name(self) -> None:
        """
        :return:
        """

        for layer in self.model.layers:
            print(layer.name)


    def _calculate_loss(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: A loaded image that has been converted into a tf.Tensor
        :return:

        Pass forward the image through the model to retrieve the activations.
        Converts the image into a batch of size 1.

        Quote from tutorial that nicely explains what we are trying to achieve on a low level:

        "The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way
        that the image increasingly "excites" the layers. The complexity of the features incorporated
        depends on layers chosen by you, i.e, lower layers produce strokes or simple patterns, while deeper
        layers give sophisticated features in images, or even whole objects."

        "Normally, loss is a quantity you wish to minimize via gradient descent. In DeepDream,
        you will maximize this loss via gradient ascent.
        """

        img_batch: np.ndarray = tf.expand_dims(image, axis=0)
        layer_activations: np.ndarray = self.model(img_batch)

        if len(layer_activations) == 1:
            layer_activations: List[np.ndarray] = [layer_activations]

        losses: List[np.ndarray] = []

        for activation in layer_activations:
            loss: np.ndarray = tf.math.reduce_mean(activation)
            losses.append(loss)

        return tf.reduce_sum(losses)


    @staticmethod
    def _random_roll(image: np.ndarray, max_roll) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param image:
        :param max_roll:
        :return:

        Randomly shift the image to avoid tiled boundaries.
        """

        shift: np.ndarray = tf.random.uniform(shape=[2], minval=-max_roll, maxval=max_roll, dtype=tf.int32)
        img_rolled: np.ndarray = tf.roll(image, shift=shift, axis=[0, 1])

        return shift, img_rolled


    @staticmethod
    def _normalize_image(input_image: np.ndarray) -> np.ndarray:
        """
        :param input_image:
        :return:

        Normalizes an image.
        """

        output_image: np.ndarray = 255 * (input_image + 1) / 2

        return tf.cast(output_image, tf.uint8)
