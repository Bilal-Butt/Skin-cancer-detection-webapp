"""
classification.py
EfficientNetB3 classifier for HAM10000 skin lesion types.

Changes from v1:
  - Upgraded backbone: EfficientNetB0 → EfficientNetB3 (+2-4% accuracy)
  - Better head: added extra Dense(256) + BatchNormalization layers
  - Masked input support: pass a segmentation mask to focus on lesion only
  - IMG_SIZE updated to 300 (EfficientNetB3 preferred input size)
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 300   # EfficientNetB3 preferred resolution (vs 256 for B0)

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


def build_classifier(img_size: int = IMG_SIZE) -> tuple:
    """
    Build EfficientNetB3 with an improved 7-class head.

    Architecture improvements:
      - EfficientNetB3 has more parameters and better feature extraction than B0
      - Added Dense(256) layer before Dense(128) for richer representations
      - BatchNormalization after each Dense to stabilise training
      - Slightly higher Dropout(0.4) to match the larger model capacity

    Returns (model, base_model) so finetune.py can selectively unfreeze layers.
    """
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False          # freeze during Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Deeper head for better feature discrimination
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(7, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def preprocess_for_classifier(
    image: Image.Image,
    mask: np.ndarray = None,
    target_size: tuple = (IMG_SIZE, IMG_SIZE)
) -> np.ndarray:
    """
    Prepare an image for EfficientNetB3.

    Args:
        image       : PIL Image (RGB)
        mask        : Optional binary mask (H x W, values 0/1) from U-Net.
                      When provided, background pixels are zeroed out so the
                      classifier focuses only on the lesion region.
        target_size : Resize target (default 300x300 for EfficientNetB3)

    Returns:
        np.ndarray of shape (1, target_size[0], target_size[1], 3), float32.
        EfficientNet normalises internally → pixels stay in 0-255 range.
    """
    image = image.resize(target_size).convert("RGB")
    image_array = np.array(image, dtype=np.float32)   # 0-255, EfficientNet handles normalisation

    if mask is not None:
        # Resize mask to match target size and apply to all 3 channels
        mask_resized = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(target_size),
            dtype=np.float32
        ) / 255.0                                       # back to 0-1 float
        mask_3ch = np.stack([mask_resized] * 3, axis=-1)
        image_array = image_array * mask_3ch            # zero-out background

    return np.expand_dims(image_array, axis=0)          # add batch dim → (1, H, W, 3)


def run_classification(image: Image.Image, model, mask: np.ndarray = None) -> dict:
    """
    Run the classifier and return a results dict.

    Args:
        image : PIL Image
        model : Compiled Keras model
        mask  : Optional binary mask from run_segmentation() — improves accuracy
                by removing background noise before classification.

    Returns dict with keys:
        predicted_class, risk_level, confidence, all_scores
    """
    image_array = preprocess_for_classifier(image, mask=mask)
    scores      = model.predict(image_array, verbose=0)[0]   # shape (7,)
    pred_idx    = int(np.argmax(scores))
    class_name, risk_level = CLASS_LABELS[pred_idx]

    return {
        "predicted_class": class_name,
        "risk_level":      risk_level,
        "confidence":      float(scores[pred_idx]),
        "all_scores": {CLASS_LABELS[i][0]: float(scores[i]) for i in range(7)},
    }
