"""
classification.py
EfficientNetB3 classifier for HAM10000 skin lesion types.

Bugs fixed:
  1. Mixed-precision policy is NO LONGER set at module import time.
     Previously `set_global_policy("mixed_float16")` ran on every import,
     which:
       (a) crashed CPU-only inference (Streamlit app, local runs);
       (b) overrode any policy app.py had already set, since module-level
           code runs at import time after the caller's own setup.
     Fix: policy setup moved into enable_mixed_precision(), called
     explicitly only by training scripts before building the model.

  2. Docstring corrected: preprocess_for_classifier returns float32, not float16.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.models import Model

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 300   # EfficientNetB3 preferred resolution

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


def enable_mixed_precision() -> str:
    """
    Enable GPU-conditional mixed precision. Call this ONCE at the start of
    a training script (e.g. finetune.py), BEFORE building any model.

    - GPU present  → 'mixed_float16'  (~40 % less VRAM, faster on T4/V100)
    - CPU only     → 'float32'        (mixed_float16 unsupported on CPU)

    Inference scripts (app.py) should NOT call this — float32 is the safe
    default and works everywhere.

    Returns the policy name that was set.
    """
    gpus   = tf.config.list_physical_devices("GPU")
    policy = "mixed_float16" if gpus else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"[classification] Precision policy set to '{policy}' "
          f"({'GPU detected' if gpus else 'no GPU — using float32'})")
    return policy


def build_classifier(img_size: int = IMG_SIZE) -> tuple:
    """
    Build EfficientNetB3 with a 7-class classification head.

    The output softmax is explicitly cast to float32. With mixed_float16
    this promotes logits from float16 → float32, preventing NaN losses.
    In float32 mode it's a no-op.

    Returns (model, base_model) so training scripts can selectively unfreeze
    base layers in Phase 2.
    """
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False   # freeze during Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(7)(x)
    x = Activation("softmax", dtype="float32")(x)   # always float32 output

    model = Model(inputs=base_model.input, outputs=x)
    return model, base_model


def build_dataset(
    image_paths: list,
    labels: list,
    batch_size: int = 16,
    augment: bool = False,
    cache_path: str = ""
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline to avoid loading all images into CPU RAM.

    Args:
        image_paths : List of file paths to images.
        labels      : List of integer class labels.
        batch_size  : Batch size (use 16 or lower on Colab free tier).
        augment     : Apply random flips / brightness for training set.
        cache_path  : Disk path to cache decoded images (e.g.
                      "/content/train_cache"). Empty string = RAM cache.
                      Use a disk path if you hit the ~12 GB Colab RAM limit.

    Returns:
        A batched, prefetched tf.data.Dataset ready for model.fit().
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def load_and_resize(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)   # [0, 255]; EfficientNet normalises internally
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(cache_path)

    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(AUTOTUNE)
    return ds


def preprocess_for_classifier(
    image: Image.Image,
    mask: np.ndarray = None,
    target_size: tuple = (IMG_SIZE, IMG_SIZE)
) -> np.ndarray:
    """
    Prepare a single PIL image for inference with EfficientNetB3.

    Args:
        image       : PIL Image (RGB).
        mask        : Optional binary mask (H×W, values 0/1) from U-Net.
                      Background pixels are zeroed out so the classifier
                      focuses only on the lesion region.
        target_size : Resize target (default (300, 300) for EfficientNetB3).

    Returns:
        np.ndarray of shape (1, H, W, 3), dtype float32, pixels in [0, 255].
    """
    image       = image.resize(target_size).convert("RGB")
    image_array = np.array(image, dtype=np.float32)   # [0, 255], float32

    if mask is not None:
        mask_resized = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(target_size),
            dtype=np.float32
        ) / 255.0
        mask_3ch    = np.stack([mask_resized] * 3, axis=-1)
        image_array = image_array * mask_3ch

    return np.expand_dims(image_array, axis=0)


def run_classification(
    image: Image.Image,
    model,
    mask: np.ndarray = None
) -> dict:
    """
    Run the classifier and return a results dict.

    Returns dict with keys:
        predicted_class, risk_level, confidence, all_scores
    """
    image_array = preprocess_for_classifier(image, mask=mask)
    scores      = model.predict(image_array, verbose=0)[0]
    pred_idx    = int(np.argmax(scores))
    class_name, risk_level = CLASS_LABELS[pred_idx]

    return {
        "predicted_class": class_name,
        "risk_level":      risk_level,
        "confidence":      float(scores[pred_idx]),
        "all_scores": {CLASS_LABELS[i][0]: float(scores[i]) for i in range(7)},
    }
