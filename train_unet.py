# =============================================================================
#  Skin Cancer Detector — U-Net Segmentation Training
#  HAM10000 dataset  |  UNET_light
#
#  Fixes:
#  - All training/evaluation code is now inside main() and guarded by
#    if __name__ == "__main__". Previously everything ran at module level,
#    so importing this file in another script would immediately start training.
#  - Removed unused `import pandas as pd`.
#  - n_imgs is now clamped to the actual test batch size to prevent
#    an IndexError when the batch has fewer than 10 images.
# =============================================================================

# ── Installs (run once in Colab) ─────────────────────────────────────────────
# !pip install kaggle scikit-learn
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d surajghuwalewala/ham1000-segmentation-and-classification
# !unzip -q ham1000-segmentation-and-classification.zip

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import glob as gb
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Conv2DTranspose, add
)
from tensorflow.keras import Model, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH_SIZE  = 16
BUFFER_SIZE = 1000
IMG_DIR     = "/content/images"    # adjust if running locally
MASK_DIR    = "/content/masks"
MODEL_DIR   = "skin-cancer-detector/models"


# =============================================================================
#  SECTION 1 — Data validation
# =============================================================================

def validate_data(img_dir, mask_dir):
    img_files  = sorted([f for f in os.listdir(img_dir)  if f.endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    image_names = [os.path.splitext(f)[0] for f in img_files]
    mask_names  = [os.path.splitext(f)[0].replace("_segmentation", "")
                   for f in mask_files]

    missing_masks = [f for f in image_names if f not in mask_names]
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} images have no mask — "
              f"{missing_masks[:5]}")
    else:
        print(f"All {len(img_files)} images have matching masks.")

    return img_files, mask_files


# =============================================================================
#  SECTION 2 — EDA visualisations
# =============================================================================

def display_image_and_mask(img_dir, mask_dir, img_files, n=5, seed=42):
    np.random.seed(seed)
    fig, axs = plt.subplots(2, n, figsize=(20, 6))
    for i in range(n):
        idx  = np.random.randint(0, len(img_files))
        img  = Image.open(os.path.join(img_dir, img_files[idx]))
        mask = Image.open(os.path.join(
            mask_dir,
            os.path.splitext(img_files[idx])[0] + "_segmentation.png"
        ))
        axs[0, i].imshow(img);  axs[0, i].set_title("Image"); axs[0, i].axis("off")
        axs[1, i].imshow(mask); axs[1, i].set_title("Mask");  axs[1, i].axis("off")
    plt.tight_layout()
    plt.show()


def display_image_with_mask(img_dir, mask_dir, img_files, n=5, seed=42):
    np.random.seed(seed)
    fig, axs = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        idx     = np.random.randint(0, len(img_files))
        img_np  = np.array(Image.open(os.path.join(img_dir, img_files[idx])))
        mask_np = np.array(Image.open(os.path.join(
            mask_dir,
            os.path.splitext(img_files[idx])[0] + "_segmentation.png"
        )))
        axs[i].imshow(img_np)
        axs[i].imshow(mask_np, cmap="Reds", alpha=0.5)
        axs[i].set_title("Image + Mask"); axs[i].axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  SECTION 3 — Data pipeline
# =============================================================================

def load_image_paths(img_dir, mask_dir):
    imgs  = sorted(gb.glob(os.path.join(img_dir,  "*.jpg")))
    masks = sorted(gb.glob(os.path.join(mask_dir, "*.png")))
    return np.array(imgs), np.array(masks)


def process(img_path, mask_path):
    img  = tf.io.read_file(img_path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img  = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0
    return img, mask


def create_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# =============================================================================
#  SECTION 4 — Model definition
# =============================================================================

def UNET_light():
    inputs       = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    filters_list = [8, 16, 32, 64]
    skips        = []

    # Stem
    x = Conv2D(8, 3, padding="same")(inputs)
    x = BatchNormalization()(x); x = Activation("relu")(x)

    # Encoder
    for f in filters_list:
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        skips.append(x)
        x = MaxPooling2D()(x)

    # Bottleneck
    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Conv2D(128, 3, padding="same")(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)

    # Decoder
    for f in reversed(filters_list):
        x    = Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        skip = skips.pop()
        x    = add([x, skip])
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x); x = Activation("relu")(x)

    outputs = Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs)


# =============================================================================
#  SECTION 5 — Loss / metrics
# =============================================================================

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f     = K.flatten(y_true)
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# =============================================================================
#  SECTION 6 — Training plots
# =============================================================================

def plot_history(history):
    metrics = [
        ("loss",             "val_loss",            "Loss"),
        ("dice_coefficient", "val_dice_coefficient", "Dice Coefficient"),
        ("accuracy",         "val_accuracy",         "Accuracy"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (train_key, val_key, title) in zip(axs, metrics):
        ax.plot(history.history[train_key], label="Train",      linestyle="--")
        ax.plot(history.history[val_key],   label="Validation")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
#  SECTION 7 — Prediction visualisation
# =============================================================================

def display_predictions(n, test_imgs, test_masks, y_pred):
    fig, axs = plt.subplots(3, n, figsize=(20, 8))
    for i in range(n):
        dice_val = dice_coefficient(test_masks[i], y_pred[i]).numpy()
        axs[0, i].imshow(test_imgs[i]);  axs[0, i].set_title("Image");
        axs[0, i].axis("off")
        axs[1, i].imshow(test_masks[i]); axs[1, i].set_title("Actual Mask")
        axs[1, i].axis("off")
        axs[2, i].imshow(y_pred[i]);
        axs[2, i].set_title(f"Pred\nDice:{dice_val:.3f}")
        axs[2, i].axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  MAIN
# =============================================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Section 1: Validate ───────────────────────────────────────────────────
    img_files, _ = validate_data(IMG_DIR, MASK_DIR)

    # ── Section 2: EDA ────────────────────────────────────────────────────────
    display_image_and_mask(IMG_DIR, MASK_DIR, img_files)
    display_image_with_mask(IMG_DIR, MASK_DIR, img_files)

    # ── Section 3: Build datasets ─────────────────────────────────────────────
    X, y = load_image_paths(IMG_DIR, MASK_DIR)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"{'Training:':<15}{len(X_train)}")
    print(f"{'Validation:':<15}{len(X_val)}")
    print(f"{'Testing:':<15}{len(X_test)}")

    train_set = create_dataset(X_train, y_train, shuffle=True)
    val_set   = create_dataset(X_val,   y_val)
    test_set  = create_dataset(X_test,  y_test)

    # ── Section 4–5: Build and compile model ──────────────────────────────────
    model = UNET_light()
    model.summary()

    model.compile(
        optimizer=Adam(0.0002),
        loss=BinaryCrossentropy(),
        metrics=["accuracy", Precision(name="precision"),
                 Recall(name="recall"), dice_coefficient]
    )

    # ── Section 6: Train ──────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        "best_weights.weights.h5",
        monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, verbose=1
    )

    history = model.fit(
        train_set, epochs=10,
        validation_data=val_set,
        callbacks=[checkpoint_cb, reduce_lr_cb]
    )

    # ── Section 7: Plots ──────────────────────────────────────────────────────
    plot_history(history)

    # ── Section 8: Evaluate ───────────────────────────────────────────────────
    model.load_weights("best_weights.weights.h5")

    loss, accuracy, precision, recall, dice = model.evaluate(test_set)
    print(f"\n{'Test Loss:':<25}{loss:.4f}")
    print(f"{'Test Accuracy:':<25}{accuracy:.4f}")
    print(f"{'Test Precision:':<25}{precision:.4f}")
    print(f"{'Test Recall:':<25}{recall:.4f}")
    print(f"{'Test Dice Coefficient:':<25}{dice:.4f}")

    # ── Section 9: Visualise predictions ─────────────────────────────────────
    test_imgs, test_masks = next(iter(test_set))
    # FIX: clamp n_imgs to actual batch size to prevent IndexError
    n_imgs = min(10, len(test_imgs))
    y_pred = model.predict(test_imgs[:n_imgs], verbose=1)
    display_predictions(n_imgs, test_imgs[:n_imgs], test_masks[:n_imgs], y_pred)

    # ── Section 10: Export TFLite ─────────────────────────────────────────────
    converter              = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model           = converter.convert()

    tflite_path = "unet_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {tflite_path}")

    shutil.copy(tflite_path, os.path.join(MODEL_DIR, "unet_model.tflite"))
    print(f"Copied to {MODEL_DIR}/unet_model.tflite")

    model.save("unet_model.h5")
    print(f"Model size (H5): {os.path.getsize('unet_model.h5') / (1024**2):.2f} MB")

    # ── FLOPs report ─────────────────────────────────────────────────────────
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2
        )
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

        full_model    = tf.function(lambda x: model(x))
        concrete_func = full_model.get_concrete_function(
            tf.TensorSpec((1, IMG_SIZE, IMG_SIZE, 3), tf.float32)
        )
        frozen_func   = convert_variables_to_constants_v2(concrete_func)
        flops_profile = profile(
            graph=frozen_func.graph,
            options=ProfileOptionBuilder.float_operation()
        )
        print(f"Total FLOPs : {flops_profile.total_float_ops:,}")
        print(f"GFLOPs      : {flops_profile.total_float_ops / 1e9:.2f}")
    except Exception as e:
        print(f"FLOPs calculation skipped ({e})")


if __name__ == "__main__":
    main()
