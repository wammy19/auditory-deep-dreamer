import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model
from typing import List, Tuple


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
        _base_model: Model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        layers: List[str] = [_base_model.get_layer(name).output for name in layer_names_for_amplification]

        # Create the feature extraction model.
        self.model = tf.keras.Model(inputs=_base_model.input, outputs=layers)


    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, _image: Tensor, tile_size: int = 512) -> Tensor:
        """
        :param _image: A loaded image that's been converted to a tf.Tensor.
        :param tile_size:
        :return:

        "Adding the gradients to the image enhances the patterns seen by the network. At each step,
        you will have created an image that increasingly excites the activations of certain layers in the network."
        """

        shift, img_rolled = self._random_roll(_image, tile_size)

        # Initialize the image gradients to zero.
        gradients: Tensor = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]

        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])

        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]

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
                    loss = self._calculate_loss(img_tile)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


    def dreamify(
            self,
            _img: Tensor,
            steps_per_octave: int = 10,
            step_size: float = 0.01,
            octaves: range = range(-2, 3),
            octave_scale: float = 1.3,
    ) -> Tensor:
        """
        :param _img:
        :param steps_per_octave:
        :param step_size:
        :param octaves:
        :param octave_scale:
        :return:

        Will create a deep dream image returned as a tf.Tensor.
        """

        _base_shape: Tensor = tf.shape(_img)
        _img = tf.keras.preprocessing.image.img_to_array(_img)
        _img = tf.keras.applications.inception_v3.preprocess_input(_img)

        initial_shape = _img.shape[:-1]
        print(initial_shape)
        _img = tf.image.resize(_img, initial_shape)

        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(_base_shape[:-1]), tf.float32) * (octave_scale ** octave)
            _img = tf.image.resize(_img, tf.cast(new_size, tf.int32))

            for step in range(steps_per_octave):
                gradients = self(_img)
                _img = _img + gradients * step_size
                _img = tf.clip_by_value(_img, -1, 1)

        result: Tensor = self._normalize_image(_img)

        return result


    def _calculate_loss(self, _image: Tensor) -> Tensor:
        """
        :param _image: A loaded image that has been converted into a tf.Tensor
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

        img_batch: Tensor = tf.expand_dims(_image, axis=0)
        layer_activations: Tensor = self.model(img_batch)

        if len(layer_activations) == 1:
            layer_activations: List[Tensor] = [layer_activations]

        losses: List[Tensor] = []

        for act in layer_activations:
            loss: Tensor = tf.math.reduce_mean(act)
            losses.append(loss)

        return tf.reduce_sum(losses)


    @staticmethod
    def _random_roll(_image: Tensor, max_roll) -> Tuple[Tensor, Tensor]:
        """
        :param _image:
        :param max_roll:
        :return:

        Randomly shift the image to avoid tiled boundaries.
        """

        shift: Tensor = tf.random.uniform(shape=[2], minval=-max_roll, maxval=max_roll, dtype=tf.int32)
        img_rolled: Tensor = tf.roll(_image, shift=shift, axis=[0, 1])

        return shift, img_rolled


    @staticmethod
    def _normalize_image(input_image: Tensor) -> Tensor:
        """
        :param input_image:
        :return:

        Normalizes an image.
        """

        output_image: Tensor = 255 * (input_image + 1) / 2

        return tf.cast(output_image, tf.uint8)
