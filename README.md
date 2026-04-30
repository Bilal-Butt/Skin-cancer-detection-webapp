# 🌸 SkinScan AI — Machine Learning Based Skin Cancer Detection

> *"Skin cancer accounts for 33.33% of all cancer cases globally. Early detection is not just helpful — it's life-saving."*
> — World Health Organization

---

## 👋 Hey, welcome!

So this is **SkinScan AI** — my final year project, and honestly one of the things I'm most proud of building.

The idea started from a pretty uncomfortable truth: skin cancer is one of the most common cancers in the world, and yet it's also one of the most *detectable* ones, if you catch it early enough. In Pakistan, Melanoma alone is the third most common skin cancer, and most people don't have easy access to a dermatologist. I kept thinking — *what if AI could help bridge that gap?* So I built this.

You upload a dermoscopy image, and the app figures out two things for you in a matter of seconds:

1. **Where exactly is the lesion?** — A U-Net model draws a pixel-level mask right around the suspicious area
2. **What type of lesion is it?** — EfficientNetB3 classifies it into one of 7 known types and tells you whether it's low, moderate, or high risk

It's not a replacement for a doctor. But it could absolutely be the thing that makes someone say *"Okay, I should probably get this checked."*

---

## 😟 Why does this even matter?

Manual examination of skin lesions is slow, expensive, and even in expert hands, prone to error. For aggressive cancers like Melanoma, the difference between catching it at Stage 1 vs Stage 3 isn't just discomfort. It's survival.

This tool was built to:
- Catch things earlier, when they're still treatable
- Give people without dermatologist access *something* to work with
- Reduce the load on clinics by pre-screening obvious cases
- Help cut down on unnecessary biopsies and procedures
- Eventually, slot into public health initiatives and mobile screening programs

---

## ✨ What it can do

| Feature | What it actually does |
|---------|----------------------|
| 🎯 **Lesion Segmentation** | U-Net draws a pixel-precise mask around the suspicious area |
| 🏷️ **Lesion Classification** | EfficientNetB3 identifies which of 7 lesion types it is |
| ⚠️ **Risk Assessment** | Automatically flags the result as High / Moderate / Low risk |
| 📊 **Confidence Scores** | Shows how confident the model is across all 7 classes |
| 🖼️ **Visual Overlay** | Side-by-side view: original image, binary mask, and a pink-red overlay |
| 👤 **Patient Information** | Capture name, age, sex, exam date, lesion location, and clinical notes |
| 🎨 **Animated UI** | Polished animated gradient hero, card-based layout, fully custom CSS |
| 🔬 **Mask-Guided Classification** | The U-Net mask is passed to the classifier to zero out background noise before it makes its call |

---

## 📊 Final Results

These are the numbers from the actual training run that produced the model files in this repo.

### Segmentation (U-Net Light)

| Metric | Test Score |
|--------|:---------:|
| **Test Accuracy** | **95.68%** |
| **Test Precision** | **94.55%** |
| **Test Recall** | **90.40%** |
| **Test Dice Coefficient** | **0.8888** |
| Test Loss (Binary CE) | 0.1104 |

Best validation Dice during training: **0.8842** at epoch 10.
Model footprint: **440K parameters · 1.32 GFLOPs · 1.7 MB after TFLite quantisation**.

### Classification (EfficientNetB3, 7 classes)

| Phase | Epochs | Best Validation Accuracy |
|-------|:------:|:-----------------------:|
| Phase 1 — head only (frozen base) | 8 | 78.12% |
| Phase 2 — fine-tune last 50 layers | 20 | **80.82%** |
| **Final test accuracy** | — | **80.14%** |

Class weights ranged from **0.21 (Nevus, the dominant class)** to **12.31 (Dermatofibroma, the rarest)** — that imbalance is the whole reason Focal Loss matters here.

---

## 🧠 How the whole thing works

### Under the hood

```
You upload a dermoscopy image
              ↓
    Resized to 256×256 for segmentation
    Resized to 300×300 for classification
              ↓
    ┌──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
U-Net (Segmentation)             EfficientNetB3 (Classification)
- Runs as a TFLite model         - Gets the image AND the mask
- Outputs a binary mask          - Background pixels zeroed out
- White pixels = lesion          - Produces 7-class probabilities
- Black pixels = healthy skin    - Highest score = predicted class
    │                                          │
    └──────────────────┬───────────────────────┘
                       ↓
              SkinScan AI Web Interface
              - Original | Mask | Overlay
              - Predicted class + risk badge
              - Confidence bar chart
              - Patient record panel
```

The part I'm most proud of here is that the segmentation and classification models actually *talk to each other*. The U-Net's output mask isn't just shown on screen — it's fed directly into EfficientNetB3 to zero out all the healthy skin, hair, and background noise before the classifier runs.

---

## 🤖 The models

### Model 1 — U-Net (Lesion Segmentation)

I built and trained a lightweight U-Net completely from scratch on the HAM10000 dataset. After training, it gets exported to TFLite so it's fast and lean enough for real-time web use.

**Architecture at a glance:**

```
Input (256×256×3)
    ↓ Stem Conv (8 filters)
    ↓ Encoder: 8 → 16 → 32 → 64 filters  (each level saves a skip connection)
    ↓ Bottleneck: 128 filters
    ↓ Decoder: Conv2DTranspose + add() to re-merge skip connections
    ↓ Output mask (256×256×1, Sigmoid)
```

| Layer Type | Count |
|-----------|-------|
| Stem Conv2D | 1 layer |
| Conv2D — Encoder | 8 layers |
| MaxPool2D | 4 layers |
| Bottleneck Conv2D | 2 layers |
| Conv2DTranspose — Upsampling | 4 layers |
| Conv2D — Decoder | 8 layers |
| Output Conv2D | 1 layer |
| **Total** | **28 layers** |

**How I trained it:**

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 2×10⁻⁴ |
| Loss | Binary Crossentropy |
| Metrics | Accuracy, Precision, Recall, Dice Coefficient |
| Batch Size | 16 |
| Epochs | 10 |
| Checkpoint | Best `val_loss` saved |
| LR Scheduler | ReduceLROnPlateau (factor=0.1, patience=5) |

> Note: Dice Coefficient is tracked as an evaluation **metric** during training, not used as a training loss. The training loss is Binary Crossentropy only.

Honestly, I was surprised by how well a "lightweight" U-Net performed. The skip connections really do make all the difference for preserving lesion boundary detail.

---

### Model 2 — EfficientNetB3 (Lesion Classification)

Rather than training a classifier from scratch — which would need millions of images — I used **transfer learning**. EfficientNetB3 was already pretrained on ImageNet (1.2 million images of real-world objects), so it already knows how to detect edges, textures, and colour gradients. I just had to teach it to apply those skills to dermoscopy images.

**Why B3 and not B0 like before?**
B3 has a richer feature hierarchy and a preferred input resolution of 300×300 (vs B0's 224×224). The original code was also forcing B0 into a non-native 256×256, which was a suboptimal compromise. Switching to B3 at its native size gives about 2–4% better accuracy without the compute cost of B4 or above.

**Improved classification head:**

```
EfficientNetB3 base (frozen in Phase 1)
    ↓ GlobalAveragePooling2D
    ↓ Dense(256, relu) + BatchNormalization + Dropout(0.4)    ← new
    ↓ Dense(128, relu) + BatchNormalization + Dropout(0.3)
    ↓ Dense(7, softmax → cast to float32 for mixed precision)
```

The extra Dense(256) layer gives the model more room to learn complex relationships between features before committing to a final class. BatchNorm keeps activations stable, and Dropout prevents the head from over-relying on any single feature — together they make the head noticeably more robust than before.

**Two-phase fine-tuning:**

| Phase | What's trained | Epochs | Initial LR | LR Schedule | Loss |
|-------|---------------|--------|------------|-------------|------|
| Phase 1 | Custom head only (base frozen) | 8 | 1×10⁻³ | Cosine Decay → 0 | Focal Loss |
| Phase 2 | Last 50 base layers + head | 20 | 3×10⁻⁵ | Cosine Decay → 0 | Focal Loss + Class Weights |

Phase 1 is about warming up — teaching the new head what skin lesion features matter, while keeping EfficientNetB3's pretrained weights completely untouched. Phase 2 is careful fine-tuning where the last 50 base layers slowly adapt their ImageNet representations toward dermoscopic patterns.

Class weights are computed with `sklearn`'s `compute_class_weight('balanced', ...)` and applied only during Phase 2. Best Phase 1 weights are reloaded before Phase 2 starts, so fine-tuning continues from peak performance, not the last epoch.

**Why Focal Loss?**
This was one of the most important changes I made. HAM10000 is badly imbalanced — Melanocytic Nevus makes up about 67% of all images. With standard categorical crossentropy, the model learns to predict NV constantly and still reports 67% accuracy — while completely missing every Melanoma. Focal Loss (γ=2.0, α=0.25) down-weights the easy majority-class examples and puts gradient pressure on the hard, rare ones. That's exactly where the clinical value lives.

**Why Cosine Decay?**
ReduceLROnPlateau is *reactive* — it only reduces the learning rate after the model has already stagnated. Cosine Decay is *proactive* — it smoothly reduces the learning rate from the very first step, so training is consistently stable rather than lurching between plateaus and sudden drops.

**Augmentation:**

| Augmentation | Value | Why it's there |
|-------------|-------|---------------|
| Random horizontal flip | — | Standard spatial invariance |
| Random vertical flip | — | Standard spatial invariance |
| Random brightness | ±0.15 | Lighting variation across dermoscopes |
| Random contrast | 0.85–1.15 | Dermoscope intensity variation |
| Random hue | ±0.05 | Skin tone variation across populations ← **new** |
| Random saturation | 0.8–1.2 | Colour cast variation in dermoscopy ← **new** |
| Pixel clip | [0, 255] | Keep values in valid range after augmentation |

The hue and saturation augmentations were a deliberate addition. If the model only trains on the specific colour distribution of HAM10000, it learns colour shortcuts instead of genuine structural features. These augmentations push it to recognise lesion morphology regardless of skin tone.

---

### Why U-Net beat everything else for segmentation

| Model | Mean Accuracy | Training Time | Memory |
|-------|--------------|---------------|--------|
| CNN (baseline) | 67.26% | 14 hours | 0.77 GB |
| DenseNet201 | 86.56% | 10 hours | 1.90 GB |
| **U-Net (ours)** | **95.68%** | **~30 min on T4** | **2.70 GB** |

U-Net trained in half an hour on a free Colab T4. The CNN took 14 hours and got 28% worse accuracy. The choice was pretty easy after that.

---

## 📂 The 7 Skin Lesion Classes

| Code | Full Name | Risk Level |
|------|-----------|------------|
| AKIEC | Actinic Keratosis | 🟡 Moderate |
| BCC | Basal Cell Carcinoma | 🔴 High |
| BKL | Benign Keratosis-like Lesion | 🟢 Low |
| DF | Dermatofibroma | 🟢 Low |
| MEL | Melanoma | 🔴 High |
| NV | Melanocytic Nevus | 🟢 Low |
| VASC | Vascular Lesion | 🟢 Low |

---

## 📊 Dataset — HAM10000

Everything was trained on the **HAM10000 (Human Against Machine with 10,000 training images)** dataset — one of the gold standards in dermatological AI research. All 10,015 images are 600×450 RGB dermoscopy photographs with expert-annotated segmentation masks and one-hot encoded classification labels.

The two models use different split ratios because segmentation has a simpler output space and generalises well with less validation data, while classification benefits from a larger held-out set to monitor per-class performance more reliably.

**Segmentation splits** (`train_unet.py` — 90 / 5 / 5):

| Split | Images |
|-------|--------|
| Training | 9,013 |
| Validation | 501 |
| Test | 501 |
| **Total** | **10,015** |

**Classification splits** (`finetune.py` — 80 / 10 / 10):

| Split | Images |
|-------|--------|
| Training | 8,012 |
| Validation | 1,001 |
| Test | 1,002 |
| **Total** | **10,015** |

---

## 🗂️ Project Structure

```
SkinScan-AI/
│
├── app.py                  ← Streamlit web interface (the SkinScan AI frontend)
├── segmentation.py         ← U-Net TFLite inference wrapper
├── classification.py       ← EfficientNetB3 inference + class labels + mask support
├── finetune.py             ← EfficientNetB3 training pipeline (Focal Loss + Cosine Decay)
├── train_unet.py           ← U-Net training pipeline
├── requirements.txt        ← Python dependencies
├── README.md               ← You're reading it :)
│
└── models/
    ├── unet_model.tflite              ← Optimized TFLite segmentation model (~1.7 MB)
    ├── efficientnet_best.weights.h5   ← Best classifier weights (~40 MB, monitored by val_accuracy)
    └── efficientnet_classifier.keras  ← Full saved classifier model (~150 MB)
```

> ⚠️ **Model files are NOT included in this repo** — they're too large for GitHub (100MB limit). Grab them from the Drive link below.

---

## 🚀 How to run it

### Option A — Google Colab with public URL via ngrok (easiest)

```python
# Cell 1 — Install dependencies
!pip install -q streamlit pyngrok

# Cell 2 — Mount Drive and copy models over
from google.colab import drive
drive.mount("/content/drive")
import shutil, os
os.makedirs("/content/skin-cancer-detector/models", exist_ok=True)
for f in ["unet_model.tflite", "efficientnet_best.weights.h5"]:
    shutil.copy(f"/content/drive/MyDrive/skin-cancer-detector-models/{f}",
                f"/content/skin-cancer-detector/models/{f}")
    print("Loaded:", f)

# Cell 3 — Set ngrok auth token (free at dashboard.ngrok.com)
from pyngrok import ngrok, conf
conf.get_default().auth_token = "PASTE_YOUR_NGROK_TOKEN_HERE"

# Cell 4 — Launch Streamlit + open public URL
import subprocess, threading, time
threading.Thread(
    target=lambda: subprocess.run([
        "streamlit", "run", "/content/skin-cancer-detector/app.py",
        "--server.port", "8501", "--server.headless", "true"
    ]),
    daemon=True
).start()
time.sleep(15)
print("✅ App live at:", ngrok.connect(8501, "http"))
```

### Option B — Run it locally

```bash
# Clone the repo
git clone https://github.com/Bilal-Butt/Skin-cancer-detection-webapp.git
cd Skin-cancer-detection-webapp

# Install dependencies
pip install -r requirements.txt

# Download models (link below) and drop them in models/

# Start the app
streamlit run app.py
```

Then open `http://localhost:8501` and you're good to go.

---

## 📥 Download the trained models

The model files live on Google Drive since they're too big for GitHub:

👉 **[Download Models — Google Drive](https://drive.google.com/drive/folders/18guFp51rYuHGRm4j1sXKn2-zqnH1fDu4?usp=drive_link)**

Once downloaded, place them here:
```
models/
├── unet_model.tflite
├── efficientnet_best.weights.h5
└── efficientnet_classifier.keras
```

---

## 📦 Requirements

```
streamlit
tensorflow >= 2.10
numpy
pandas
pillow
matplotlib
scikit-learn
pyngrok
```

```bash
pip install -r requirements.txt
```

---

## 💡 Decisions I made and why

**Why TFLite for segmentation?**
The full Keras U-Net is great for training, but it's overkill for inference. TFLite strips out everything that isn't needed at runtime — the result is a model that's faster to load, smaller in memory (1.7 MB vs 5.3 MB), and runs comfortably inside a Streamlit app without hogging resources.

**Why EfficientNetB3 over B0?**
More expressive feature hierarchy, higher preferred input resolution (300×300 vs 224×224), and 2–4% better accuracy on fine-grained texture tasks — all without the compute cost of B4 or B5. It's the sweet spot for this kind of work. The original setup was also forcing B0 into a non-native 256×256, which was a silent accuracy penalty fixed by moving to B3 at its native resolution.

**Why Focal Loss over plain crossentropy?**
Because HAM10000 lies to you if you let it. The dataset is dominated by Melanocytic Nevus at ~67%. A model trained with standard crossentropy learns to predict NV constantly and still reports 67% accuracy — while completely missing every Melanoma. Focal Loss (γ=2.0, α=0.25) forces the model to focus on the hard examples it keeps getting wrong, which is exactly where clinical value lives.

**Why Cosine Decay instead of ReduceLROnPlateau alone?**
ReduceLROnPlateau only kicks in *after* training has already stalled. Cosine Decay starts reducing the learning rate smoothly from epoch 1, so you never get the sharp cliffs that purely reactive schedulers create. A ReduceLROnPlateau callback is still kept as a safety net, but Cosine Decay does the heavy lifting.

**Why pass the segmentation mask into the classifier?**
Most pipelines dump the whole image into the classifier and hope it figures out what to focus on. But dermoscopy images are noisy — healthy skin, hair, ruler artefacts, and lighting gradients are all competing for attention. By zeroing out everything outside the U-Net mask first, the classifier gets a clean, lesion-only input. It's a one-line change in `preprocess_for_classifier()` that makes a real difference.

**Why GPU-conditional mixed precision?**
The original code set `mixed_float16` at module import time, which crashed CPU-only inference (i.e. running the Streamlit app on a laptop). The fix moves precision setup into `enable_mixed_precision()`, called explicitly only by training scripts. Inference always runs in float32 — safe everywhere, GPU or not. The output softmax is also explicitly cast back to float32 to prevent NaN losses with mixed precision.

**Why skip connections in U-Net?**
Downsampling compresses spatial information — that's the whole point of the encoder. But for segmentation you actually *need* that fine-grained spatial detail to draw an accurate mask. Skip connections route the high-resolution encoder feature maps directly to the decoder at each scale, so the model can reconstruct clean lesion boundaries even after aggressive max-pooling. The decoder uses `add()` (residual-style merging) rather than concatenation, keeping the parameter count low.

**Why hue and saturation augmentation?**
Skin tone varies enormously across the global population. If you train only on the colour distribution of HAM10000, the model learns colour shortcuts instead of genuine structural features. Random hue (±0.05) and saturation (0.8–1.2) shifts during training force the model to generalise across skin tones. Pixels are correctly scaled to [0, 1] for those ops then back to [0, 255] for EfficientNet — a subtle but critical detail, since `tf.image.random_hue` silently mangles inputs in the wrong range.

---

## 🖥️ What I trained on

| Component | Spec |
|-----------|------|
| Processor | Core i7, 2.30 GHz |
| Training Environment | Google Colab (Tesla T4 GPU, 15 GB VRAM) |
| Language | Python 3.12 |
| Framework | TensorFlow / Keras |
| Total training time | ~3.5 hours (segmentation + 2-phase classification) |

---

## 🌍 Why this matters beyond the code

**For people:**
- Skin cancer screening is expensive and inaccessible in a lot of places. This makes it a bit less so.
- Self-screening from home becomes possible before ever stepping into a clinic.
- Awareness matters. If seeing "High Risk — Melanoma" makes someone book an appointment they'd otherwise skip, that's a life.

**For healthcare:**
- Doctors are overloaded. Pre-screening tools that filter out obvious benign cases give them more time for the complex ones.
- Fewer unnecessary biopsies means lower costs and less stress for patients.
- The patient record capture feature means this can slot into real clinical workflows, not just live as a demo.

---

## 🛣️ What's next

Things I'd like to add when I get the time:

- [ ] Test-time augmentation (TTA) — usually adds 1–2% accuracy for free
- [ ] Grad-CAM visualisations to show *why* the model made a call
- [ ] ONNX export for cross-framework deployment
- [ ] Multi-modal input — combine the image with patient metadata (age, sex, lesion location)
- [ ] Docker container for one-command deployment
- [ ] FastAPI backend so this can power mobile apps too

---

## 📚 References

1. WHO — Global skin cancer burden statistics
2. Global Cancer Statistics 2020 — Melanoma in Pakistan
3. Ronneberger O., Fischer P., Brox T. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI
4. Tan M., Le Q. (2019) — *EfficientNet: Rethinking Model Scaling for CNNs*, Google Brain
5. Lin T., et al. (2017) — *Focal Loss for Dense Object Detection*, Facebook AI Research
6. Tschandl P., Rosendahl C., Kittler H. (2018) — *HAM10000 Dataset*
7. ISIC Archive — International Skin Imaging Collaboration
8. Aksoy S. (2025) — *Skin Lesion Segmentation Using U-Net*, IJFMR vol. 7

---

## ⚠️ Disclaimer

> This tool is built for **educational and research purposes only**. It is **not a certified medical device** and should **never replace professional medical advice, diagnosis, or treatment**. Please — if something looks wrong, see a dermatologist. This app can point you in a direction. A qualified doctor makes the call.

---

## 👨‍💻 About me

**Muhammad Bilal Butt**
- 🎓 BS Computer Engineering — COMSATS University Islamabad, Abbottabad Campus
- 💻 GitHub: [@Bilal-Butt](https://github.com/Bilal-Butt)
- 📧 Email: mics.pes@gmail.com

I built this as my final year project, but it became something I genuinely care about. If you're working on something similar, or have feedback, I'd love to hear from you.

---

## 📄 License

Open source under the [MIT License](LICENSE). Use it, build on it, improve it.

---

*If this project was useful to you or sparked an idea — a ⭐ on GitHub genuinely means a lot. It helps other people find it too.*
