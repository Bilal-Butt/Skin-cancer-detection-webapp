"""
segmentation.py
TFLite inference wrapper for the UNET_light segmentation model.

Fixes:
  - run_segmentation() now accepts an optional pre-loaded interpreter.
    Previously it always reloaded the TFLite model from disk on every call,
    even when app.py had already cached the interpreter via load_models().
    Now app.py can pass its cached interpreter and avoid redundant I/O.
"""

import numpy as np
from PIL import Image
import tensorflow as tf


def load_segmentation_model(model_path: str = "models/unet_model.tflite") -> tf.lite.Interpreter:
    """Load the TFLite model and allocate tensors."""
    if not tf.io.gfile.exists(model_path):
        raise FileNotFoundError(
            f"TFLite model not found at '{model_path}'.\n"
            "Run train_unet.py first to generate unet_model.tflite, "
            "then place it in the models/ folder next to app.py."
        )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: Image.Image, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Resize, convert to RGB, normalise to [0, 1], and add batch dimension.
    Returns array of shape (1, 256, 256, 3), float32.
    """
    image       = image.resize(target_size).convert("RGB")
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

    mask = interpreter.get_tensor(output_details[0]["index"])   # (1, 256, 256, 1)
    mask = mask.squeeze()                                        # (256, 256)
    return (mask > 0.5).astype(np.uint8)


def run_segmentation(
    image: Image.Image,
    model_path: str = "models/unet_model.tflite",
    interpreter: tf.lite.Interpreter = None
) -> np.ndarray:
    """
    End-to-end helper: preprocess image, run segmentation, return binary mask.

    Args:
        image       : PIL Image to segment.
        model_path  : Path to the TFLite model file. Used only when
                      `interpreter` is None.
        interpreter : Optional pre-loaded TFLite interpreter. Pass this to
                      avoid reloading the model from disk on every call.
                      In app.py, pass the cached interpreter from load_models().

    Returns:
        Binary mask as np.ndarray of shape (256, 256), dtype uint8.
    """
    if interpreter is None:
        interpreter = load_segmentation_model(model_path)
    image_array = preprocess_image(image)
    return predict_mask(interpreter, image_array)
