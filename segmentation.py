"""
segmentation.py
TFLite inference wrapper for the UNET_light segmentation model.
"""

import numpy as np
from PIL import Image
import tensorflow as tf


def load_segmentation_model(model_path: str = "models/unet_model.tflite") -> tf.lite.Interpreter:
    """Load the TFLite model and allocate tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: Image.Image, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Resize, convert to RGB, normalise to [0, 1], and add batch dimension.
    Returns array of shape (1, 256, 256, 3).
    """
    image = image.resize(target_size).convert("RGB")
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_mask(interpreter: tf.lite.Interpreter, image_array: np.ndarray) -> np.ndarray:
    """
    Run inference and return a binary mask of shape (256, 256).
    Pixels > 0.5 are classified as lesion.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()

    mask = interpreter.get_tensor(output_details[0]["index"])  # (1, 256, 256, 1)
    mask = mask.squeeze()                                       # (256, 256)
    return (mask > 0.5).astype(np.uint8)


def run_segmentation(image: Image.Image, model_path: str = "models/unet_model.tflite") -> np.ndarray:
    """End-to-end helper: load model, preprocess, predict, return binary mask."""
    interpreter  = load_segmentation_model(model_path)
    image_array  = preprocess_image(image)
    return predict_mask(interpreter, image_array)
