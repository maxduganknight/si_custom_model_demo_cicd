"""Library Import."""
import tensorflow as tf

from .utility import get_masks, plotter


class Postprocessor:
    """Postprocessor class."""

    def run(
        self,
        preprocessed_image_as_bytes: bytes,
        logits_as_bytes: bytes,
        threshold=0.5,
        disclaimer_text="",
    ):
        """_summary_.

        Args:
            preprocessed_image_as_bytes (bytes): preprocessed image encoded via base64
            logits_as_bytes (bytes): output logit encoded via base64 obtained from the model
            threshold (float, optional): prediction threshold. Defaults to 0.5.
            disclaimer_text (str, optional): disclaimer_text. Defaults to "".

        Returns:
            all_segments (nparray): segmented image array including all regions of arm bones
            int1 (nparray): Segmented Intraarticular_1
            prox (nparray): Segmented Proximal_1/3
            mid (nparray): Segmented Mid_1/3
            dist (nparray): Segmented Distal_1/3
            int2 (nparray): Segmented Intraarticular_2
        """
        assert (
            len(disclaimer_text) < 31
        ), f"disclaimer text {disclaimer_text} is longer than 30 characters"

        # In the production phase the preprocessed_image_as_bytes should be decoded via the base64 to numpy array
        # using img = base64.b64decode(preprocessed_image_as_bytes)
        # and then we get the numpy array via input_data = np.array(imageio.imread(img)).astype(np.float32)

        input_data = preprocessed_image_as_bytes
        input_data = input_data[0]

        # In the production phase the logits_as_bytes shoild be decoded via the base64 to numpy array
        # using logits = base64.b64decode(logits_as_bytes)
        # and then we get the numpy array via logits = np.array(imageio.imread(logits)).astype(np.float32)

        logits = logits_as_bytes

        prob = tf.sigmoid(logits)
        prediction = tf.cast(prob > threshold, dtype=tf.float32)
        output = tf.cast(prediction, dtype=tf.float32).numpy().squeeze()

        (int1_mask, prox_mask, mid_mask, dist_mask, int2_mask) = get_masks(
            output
        )  # regions are 'Intraarticular', 'Proximal_1/3', 'Mid_1/3', 'Distal_1/3', 'Intraarticular'
        all_masks = int1_mask + prox_mask + mid_mask + dist_mask + int2_mask

        all_segments = plotter(input_data, all_masks)
        int1 = plotter(input_data, int1_mask)
        prox = plotter(input_data, prox_mask)
        mid = plotter(input_data, mid_mask)
        dist = plotter(input_data, dist_mask)
        int2 = plotter(input_data, int2_mask)

        return (all_segments, int1, prox, mid, dist, int2)
