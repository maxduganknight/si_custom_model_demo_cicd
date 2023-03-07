"""Library Import."""
import numpy as np
import tensorflow as tf


class Preprocessor:
    """Preprocessor class."""

    def run(self, image_as_bytes: bytes, input_dimension=512):
        """_summary_.

        Args:
            image_as_bytes (bytes): image encoded via base64
            input_dimension (int): description for resizing the input X-ray

        Returns:
            input_data (numpy.ndarray): preprocessed image
        """
        # In the production phase the image_as_bytes should be decoded via the base64 to numpy array
        # using img = base64.b64decode(image_as_bytes)
        # and then we get the numpy array via input_data = np.array(imageio.imread(img)).astype(np.float32)

        input_data = image_as_bytes

        input_data = tf.image.resize_with_pad(
            np.expand_dims(input_data, axis=-1),
            input_dimension,
            input_dimension,
            method=tf.image.ResizeMethod.BILINEAR,
        ).numpy()
        input_data = input_data[np.newaxis, ...]

        return input_data
