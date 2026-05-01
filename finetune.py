"""
finetune.py
Two-phase EfficientNetB3 fine-tuning on the HAM10000 classification task.

Bugs fixed:
  1. Mixed precision is now activated explicitly via enable_mixed_precision()
     because classification.py no longer sets it at import time.

  2. Auto-detects dataset location (Colab /content vs Kaggle /kaggle/input).
     Previously DATASET_PATH was a hardcoded Kaggle string and IMG_DIR /
     CSV_PATH were computed at import time, so patching after import did
     not work.

  3. augment() pixel range bug: tf.image.random_hue and random_saturation
     EXPECT pixels in [0, 1]. The old code passed [0, 255] images, then
     clipped output to [0, 255] — but TF internally converts to HSV and
     clips back to [0, 1], silently destroying every pixel. Fix: scale
     to [0, 1] for hue/saturation, then scale back.

  4. Phase 2 now loads the best Phase 1 checkpoint before fine-tuning,
     instead of continuing from the last (possibly worse) Phase 1 epoch.

  5. Phase 2 fit() uses initial_epoch=8 so epoch numbers stay continuous
     in logs and checkpoints (9–28 instead of restarting at 1).

Phase 1 (8 epochs,  LR cosine 1e-3→0)  — train custom head only, base frozen.
Phase 2 (20 epochs, LR cosine 3e-5→0)  — unfreeze last 50 base layers,
                                          focal loss + class-weighted.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Enable mixed precision BEFORE importing build_classifier so the policy is
# active when EfficientNetB3 layers are constructed.
from classification import enable_mixed_precision
enable_mixed_precision()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from classification import build_classifier   # reuse architecture

# ── Auto-detect dataset path (Colab or Kaggle) ───────────────────────────────
_COLAB_ROOT  = "/content"
_KAGGLE_ROOT = "/kaggle/input/ham1000-segmentation-and-classification"

if os.path.isdir(os.path.join(_COLAB_ROOT, "images")):
    DATASET_PATH = _COLAB_ROOT
elif os.path.isdir(os.path.join(_KAGGLE_ROOT, "images")):
    DATASET_PATH = _KAGGLE_ROOT
else:
    DATASET_PATH = _COLAB_ROOT
    print("[finetune.py] WARNING: dataset not found at Colab or Kaggle paths. "
          "Set DATASET_PATH manually before calling train().")

IMG_DIR  = os.path.join(DATASET_PATH, "images")
CSV_PATH = os.path.join(DATASET_PATH, "GroundTruth.csv")

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_DIR    = "/content/skin-cancer-detector"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "efficientnet_best.weights.h5")
KERAS_PATH   = os.path.join(MODEL_DIR, "efficientnet_classifier.keras")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 300
BATCH_SIZE    = 32
CLASS_COLS    = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
PHASE1_EPOCHS = 8
PHASE2_EPOCHS = 20


# =============================================================================
#  Focal Loss
# =============================================================================

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss for multi-class classification."""
    def loss_fn(y_true, y_pred):
        y_pred       = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce           = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t          = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = alpha * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn


# =============================================================================
#  Data loading
# =============================================================================

def load_and_split_data():
    """Load CSV, build file paths, and split into train / val / test."""
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"GroundTruth.csv not found at {CSV_PATH}.\n"
            "On Colab, run:\n"
            "  !kaggle datasets download -d surajghuwalewala/ham1000-segmentation-and-classification\n"
            "  !unzip -q ham1000-segmentation-and-classification.zip"
        )

    df = pd.read_csv(CSV_PATH)
    df["filepath"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, x + ".jpg"))

    missing = ~df["filepath"].apply(os.path.isfile)
    if missing.any():
        print(f"[WARNING] {missing.sum()} image files missing — dropping them.")
        df = df[~missing].reset_index(drop=True)

    labels = df[CLASS_COLS].values
    paths  = df["filepath"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def process_image(filepath, label):
    """Load and resize. EfficientNet normalises internally so no /255."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32), label


def augment(img, label):
    """
    Augmentation with hue + saturation jitter for skin tone variation.

    BUG FIX: tf.image.random_hue and random_saturation expect input in
    [0, 1]. The previous version passed [0, 255] images, which got mangled
    inside the HSV conversion. Now we scale to [0, 1] for those ops, then
    scale back to [0, 255] for the rest of the pipeline.
    """
    # Geometric / intensity augmentations — operate on [0, 255] safely.
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15 * 255.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)

    # Colour augmentations — REQUIRE [0, 1] range.
    img = img / 255.0
    img = tf.image.random_hue(img, 0.05)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img * 255.0

    return img, label


def create_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =============================================================================
#  Training
# =============================================================================

def get_callbacks(weights_path):
    checkpoint = ModelCheckpoint(
        weights_path, monitor="val_accuracy",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-7
    )
    return [checkpoint, reduce_lr]


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    train_set = create_dataset(X_train, y_train, training=True)
    val_set   = create_dataset(X_val,   y_val)
    test_set  = create_dataset(X_test,  y_test)

    # Class weights to handle HAM10000 imbalance (Melanocytic Nevus dominates).
    y_train_idx       = np.argmax(y_train, axis=1)
    class_weights     = compute_class_weight("balanced",
                                             classes=np.unique(y_train_idx),
                                             y=y_train_idx)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", {k: f"{v:.2f}" for k, v in class_weight_dict.items()})

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    model, base_model = build_classifier(img_size=IMG_SIZE)

    steps_p1    = max(1, len(X_train) // BATCH_SIZE)
    lr_sched_p1 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=steps_p1 * PHASE1_EPOCHS
    )
    model.compile(
        optimizer=Adam(lr_sched_p1),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )

    print(f"\n── Phase 1: training classification head ({PHASE1_EPOCHS} epochs) ──")
    model.fit(
        train_set,
        epochs=PHASE1_EPOCHS,
        validation_data=val_set,
        callbacks=get_callbacks(WEIGHTS_PATH)
    )

    # FIX: load best Phase 1 weights before Phase 2.
    print(f"\nLoading best Phase 1 weights from {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    print(f"Trainable layers in Phase 2: {sum(l.trainable for l in model.layers)}")

    steps_p2    = max(1, len(X_train) // BATCH_SIZE)
    lr_sched_p2 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-5,
        decay_steps=steps_p2 * PHASE2_EPOCHS
    )
    model.compile(
        optimizer=Adam(lr_sched_p2),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )

    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS
    print(f"\n── Phase 2: fine-tuning last 50 layers "
          f"(epochs {PHASE1_EPOCHS + 1}–{total_epochs}, class-weighted) ──")
    model.fit(
        train_set,
        epochs=total_epochs,
        initial_epoch=PHASE1_EPOCHS,    # FIX: continuous epoch numbering
        validation_data=val_set,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(WEIGHTS_PATH)
    )

    # ── Final evaluation ─────────────────────────────────────────────────────
    model.load_weights(WEIGHTS_PATH)
    loss, accuracy = model.evaluate(test_set)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    model.save(KERAS_PATH)
    print(f"Model saved → {KERAS_PATH}")
    return model


if __name__ == "__main__":
    train()
