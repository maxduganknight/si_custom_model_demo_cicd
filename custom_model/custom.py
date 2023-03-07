import os
import imageio
import tensorflow as tf
import cv2
import numpy as np
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from inference import ChestONNXModel

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(input_dir):
    if not os.path.exists(os.path.join(current_dir, "inference/app_model/model.onnx")):
        inference_dir = os.path.join(current_dir, "inference")
        raise ValueError(
            "Saved Model does not exists. Please Download the app_model folder to {}.".format(
                inference_dir
            )
        )
    model = ChestONNXModel()
    return model

def score_unstructured(model, data, query, **kwargs):
    """
    Take in image as stream of bytes, apply preproccessing, 
    inference, post-processing and return binary data.
    """
    preprocessor = Preprocessor()
    postprocessor = Postprocessor()
    reader = imageio.get_reader(data)
    input_data = reader.get_data(0)
    input_data = np.array(input_data)
    preprocessed_input_data = preprocessor.run(input_data)
    logits = model.run(preprocessed_input_data)
    (all_segments, int1, prox, mid, dist, int2) = postprocessor.run(
        preprocessed_input_data, logits
    )
    is_success, buffer = cv2.imencode(".jpg", all_segments)
    result = buffer.tobytes()
    ret_kwargs = {"mimetype": "application/octet-stream"}
    ret = result, ret_kwargs
    return ret
