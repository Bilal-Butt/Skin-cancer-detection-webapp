"""
finetune.py
Two-phase EfficientNetB0 fine-tuning on the HAM10000 classification task.

Phase 1 (5 epochs, LR=1e-3)   — train the custom head only, base frozen.
Phase 2 (15 epochs, LR=5e-5)  — unfreeze last 50 base layers, class-weighted.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from classification import build_classifier   # reuse architecture definition

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "/content"
IMG_DIR      = os.path.join(DATASET_PATH, "images")
CSV_PATH     = os.path.join(DATASET_PATH, "GroundTruth.csv")
MODEL_DIR    = "skin-cancer-detector/models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "efficientnet_best.weights.h5")
KERAS_PATH   = os.path.join(MODEL_DIR, "efficientnet_classifier.keras")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH_SIZE  = 32
CLASS_COLS  = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]


# =============================================================================
#  Data loading
# =============================================================================

def load_and_split_data():
    df = pd.read_csv(CSV_PATH)
    df["filepath"] = df["image"].apply(lambda x: os.path.join(IMG_DIR, x + ".jpg"))
    labels = df[CLASS_COLS].values         # one-hot, shape (N, 7)
    paths  = df["filepath"].values

    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.2,  random_state=42)
    X_val,   X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def process_image(filepath, label):
    """Load and resize; EfficientNet normalises internally so no /255."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32), label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
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
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )
    return [checkpoint, reduce_lr]


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    train_set = create_dataset(X_train, y_train, training=True)
    val_set   = create_dataset(X_val,   y_val)
    test_set  = create_dataset(X_test,  y_test)

    # Class weights to handle HAM10000 imbalance (Melanocytic Nevus dominates)
    y_train_idx = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train_idx), y=y_train_idx)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", {k: f"{v:.2f}" for k, v in class_weight_dict.items()})

    # ── Phase 1: head only ────────────────────────────────────────────────────
    model, base_model = build_classifier()
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n── Phase 1: training classification head (5 epochs) ──")
    model.fit(
        train_set, epochs=5,
        validation_data=val_set,
        callbacks=get_callbacks(WEIGHTS_PATH)
    )

    # ── Phase 2: fine-tune last 50 base layers with class weights ─────────────
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    print(f"Trainable layers: {sum(l.trainable for l in model.layers)}")

    model.compile(
        optimizer=Adam(5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n── Phase 2: fine-tuning last 50 layers (15 epochs, class-weighted) ──")
    model.fit(
        train_set, epochs=15,
        validation_data=val_set,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(WEIGHTS_PATH)
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_weights(WEIGHTS_PATH)   # restore best checkpoint
    loss, accuracy = model.evaluate(test_set)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    model.save(KERAS_PATH)
    print(f"Model saved → {KERAS_PATH}")
    return model


if __name__ == "__main__":
    train()
