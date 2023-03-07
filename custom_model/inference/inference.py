"""Library Import."""
import os

from .model import CXRONNXModel


class ChestONNXModel:
    """Inference class."""

    def __init__(self, log_severity_level=2):
        """_summary_.

        Args:
            log_severity_level (int, optional): Defaults to 2.

        Raises:
            FileNotFoundError: FileNotFoundError
        """
        current_dir = os.path.dirname(__file__)
        models_dir = f"{current_dir}/app_model"

        self.cxr_app_paths = f"{models_dir}/model.onnx"

        # check if all the onnx models exist in the file system
        if not os.path.isfile(self.cxr_app_paths):
            raise FileNotFoundError(f"Cannot find the cxr model {self.cxr_app_paths}")

        self.cxr_apps = CXRONNXModel(self.cxr_app_paths, log_severity_level)

    def run(self, preprocessed_image_as_bytes: bytes):
        """_summary_.

        Args:
            preprocessed_image_as_bytes (bytes): preprocessed image encoded via base64

        Returns:
            logits (nparray): model's output
        """
        logits = self.cxr_apps.run(preprocessed_image_as_bytes)

        return logits
