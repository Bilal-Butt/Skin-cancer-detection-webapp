"""
classification.py
EfficientNetB0 classifier for HAM10000 skin lesion types.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# 7 HAM10000 lesion classes  →  (display name, risk level)
CLASS_LABELS = {
    0: ("Actinic Keratosis",    "Moderate"),
    1: ("Basal Cell Carcinoma", "High"),
    2: ("Benign Keratosis",     "Low"),
    3: ("Dermatofibroma",       "Low"),
    4: ("Melanoma",             "High"),
    5: ("Melanocytic Nevus",    "Low"),
    6: ("Vascular Lesion",      "Low"),
}


def build_classifier(img_size: int = 256) -> tuple:
    """
    Build EfficientNetB0 with a custom 7-class head.

    Returns (model, base_model) so the caller can selectively unfreeze
    base_model layers for fine-tuning.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False          # freeze during Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(7, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def preprocess_for_classifier(image: Image.Image, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Prepare an image for EfficientNetB0.
    EfficientNet normalises internally, so pixels should be 0-255 (float32).
    Returns array of shape (1, 256, 256, 3).
    """
    image = image.resize(target_size).convert("RGB")
    image_array = np.array(image, dtype=np.float32)   # no /255 — EfficientNet handles it
    return np.expand_dims(image_array, axis=0)


def run_classification(image: Image.Image, model) -> dict:
    """
    Run the classifier and return a results dict containing:
      - predicted_class  : str
      - risk_level       : str  ("Low" | "Moderate" | "High")
      - confidence       : float
      - all_scores       : dict  {class_name: score}
    """
    image_array = preprocess_for_classifier(image)
    scores      = model.predict(image_array, verbose=0)[0]       # shape (7,)
    pred_idx    = int(np.argmax(scores))
    class_name, risk_level = CLASS_LABELS[pred_idx]

    return {
        "predicted_class": class_name,
        "risk_level":      risk_level,
        "confidence":      float(scores[pred_idx]),
        "all_scores": {CLASS_LABELS[i][0]: float(scores[i]) for i in range(7)},
    }
