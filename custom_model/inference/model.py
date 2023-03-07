"""Library Import."""
import onnxruntime as rt


class CXRONNXModel:
    """Model."""

    def __init__(self, app_path, log_severity_level=2):
        """Load and run a deep learning model in ONNX format.

        Args:
            app_path (str): full path to the onnx model.
            log_severity_level (int): the log_severity_level property in onnxruntime.SessionOptions, default to Warning.
        Attributes:
            session (onnxruntime.InferenceSession): An onnx inference session instance.
            input_name (str): Input name of the model.
        """
        if type(app_path) is not str:
            raise TypeError("app_path must be a string")
        if not app_path.endswith(".onnx"):
            raise ValueError("Not an ONNX model")

        sess_options = rt.SessionOptions()
        sess_options.log_severity_level = log_severity_level

        self.session = rt.InferenceSession(app_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def run(self, preprocessed_image_as_bytes: bytes):
        """_summary_.

        Args:
            preprocessed_image_as_bytes (bytes): preprocessed image encoded via base64

        Returns:
            output_data (nparray): model's output
        """
        # # decode the base64 jpeg to numpy array
        # img = base64.b64decode(preprocessed_image_as_bytes)
        # input_data = np.array(imageio.imread(img)).astype(np.float32)

        input_data = preprocessed_image_as_bytes

        output_data = self.session.run(None, {self.input_name: input_data})[0]
        return output_data
