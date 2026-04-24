# 🔬 Title - Machine Learning Based Skin Cancer Detection

> *"Skin cancer accounts for 33.33% of all cancer cases globally. Early detection is not just helpful — it's life-saving."*
> — World Health Organization

---

## 👋 What Is This?

Hey! Welcome to my final year project **Machine Learning Based Skin Cancer Detection**, a deep learning powered web application that helps identify and analyze skin lesions from dermoscopy images.

This project was born out of a simple but important question: *what if AI could help catch skin cancer before it's too late?* In Pakistan alone, Melanoma is the third most common skin cancer, and access to dermatologists is nowhere near equal across the country. This tool is a step toward bridging that gap.

You upload a dermoscopy image, and the app does two things in seconds:

1. **Segments the lesion** — draws a precise mask around the suspicious area using a U-Net model
2. **Classifies the lesion** — identifies which of 7 types it is and flags the risk level using EfficientNetB0

---

## 🎯 The Problem We're Solving

Traditional manual examination of skin lesions is time-consuming and prone to human error. Delayed or inaccurate diagnoses can be the difference between life and death, especially for aggressive cancers like Melanoma, which spreads rapidly if not caught early.

This project addresses that by building an automated, accurate, and efficient AI-assisted screening tool that:
- Reduces human error in early-stage detection
- Enables faster screening with reduced treatment times
- Can be integrated into public health initiatives and self-examination workflows
- Cuts down the need for unnecessary surgeries and costly procedures

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🎯 **Lesion Segmentation** | U-Net draws a pixel-precise mask around the lesion area |
| 🏷️ **Lesion Classification** | EfficientNetB0 identifies which of 7 lesion types it is |
| ⚠️ **Risk Assessment** | Automatically flags High / Moderate / Low risk |
| 📊 **Confidence Scores** | Shows model confidence across all 7 classes |
| 🖼️ **Visual Overlay** | Side-by-side: original image, mask, and red overlay |
| 👤 **Patient Information** | Capture patient name, age, gender, date, and observations |
| 📋 **Severity Scoring** | Segmentation score with severity interpretation |

---

## 🧠 How It Works

### The Pipeline

```
Patient uploads dermoscopy image
              ↓
    Image resized to 256×256
    Pixels normalized 0.0–1.0
              ↓
    ┌─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
U-Net (Segmentation)        EfficientNetB0 (Classification)
- Binary mask output        - 7-class probability output
- White = lesion            - Highest score = predicted class
- Black = background        - Risk level assigned
    │                                 │
    └──────────────┬──────────────────┘
                   ↓
         SkinScan AI Web Interface
         - Original | Mask | Overlay
         - Predicted class + risk level
         - Confidence bar chart
         - Severity interpretation
```

---

## 🤖 The Models

### Model 1 — U-Net (Lesion Segmentation)

A lightweight custom U-Net trained from scratch on the HAM10000 dataset.

**Architecture:**

```
Input (256×256×3)
    ↓ Encoder: 8 → 16 → 32 → 64 filters
    ↓ Bottleneck: 128 filters
    ↓ Decoder with skip connections
    ↓ Output mask (256×256×1)
```

| Layer Type | Count |
|-----------|-------|
| Conv2D — Encoder | 8 layers |
| MaxPool2D | 4 layers |
| Bottleneck Conv2D | 2 layers |
| Conv2DTranspose — Upsampling | 4 layers |
| Conv2D — Decoder | 8 layers |
| Output Conv2D | 1 layer |
| **Total Layers** | **28** |

**Training Config:**

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 2×10⁻⁴ |
| Activation | ReLU + Sigmoid |
| Batch Size | 32 |
| Loss | Binary Crossentropy + Dice Loss |

**Results Achieved:**

| Metric | Score |
|--------|-------|
| Train Accuracy | 95.88% |
| Validation Accuracy | 95.19% |
| Dice Coefficient | 89.05% |
| Validation Dice | 88.28% |
| Mean Accuracy | 94.42% |
| F1 Score | 93.00% |
| Precision | 90.52% |
| Recall | 94.73% |

---

### Model 2 — EfficientNetB0 (Lesion Classification)

Instead of training from zero, we used **transfer learning** — taking EfficientNetB0 pretrained on ImageNet (1.2 million images) and fine-tuning it specifically on skin lesion data.

**Why EfficientNetB0?** It delivers the best accuracy-to-parameters ratio of any CNN architecture, making it fast enough for real-time inference while staying highly accurate.

**Two-Phase Fine-Tuning Strategy:**

- **Phase 1** — Freeze EfficientNet entirely, train only the custom classification head (5 epochs, lr = 10⁻³)
- **Phase 2** — Unfreeze the last 50 layers, fine-tune carefully at a very low learning rate (15 epochs, lr = 5×10⁻⁵)
- **Class weights** applied throughout to handle HAM10000's heavy class imbalance

---

### Model Comparison — Why We Chose U-Net

We tested three deep learning architectures during development:

| Model | Mean Accuracy | Training Time | Memory |
|-------|--------------|---------------|--------|
| CNN (baseline) | 67.26% | 14 hours | 0.77 GB |
| DenseNet201 | 86.56% | 10 hours | 1.90 GB |
| **U-Net (ours)** | **94.42%** | **2 hours** | **2.70 GB** |

U-Net won on both accuracy and training efficiency — faster to train, significantly more accurate.

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

We trained on the **HAM10000 (Human Against Machine with 10,000 training images)** dataset — one of the most widely used benchmarks in dermatological AI research.

| Split | Images |
|-------|--------|
| Training | 9,013 |
| Testing | 1,002 |
| **Total** | **10,015** |

All images are 600×450 RGB dermoscopy photographs with expert-annotated segmentation masks and one-hot encoded classification labels.

**Data Augmentation applied during training:**
- Random horizontal and vertical flips
- Random brightness and contrast adjustments
- Random zoom (±20%)
- Random rotation (up to 20°)

---

## 🗂️ Project Structure

```
SkinScan-AI/
│
├── app.py                  ← Streamlit web interface (main entry point)
├── segmentation.py         ← U-Net TFLite inference functions
├── classification.py       ← EfficientNet inference + class labels
├── finetune.py             ← EfficientNet training pipeline
├── train_unet.py           ← U-Net training pipeline
├── requirements.txt        ← Python dependencies
├── README.md               ← You are here :)
│
└── models/
    ├── unet_model.tflite              ← Optimized segmentation model
    ├── efficientnet_best.weights.h5   ← Best classifier weights
    └── efficientnet_classifier.keras  ← Full classifier model
```

> ⚠️ **Model files are NOT included in this repo** — they exceed GitHub's 100MB file size limit. Download them from the Google Drive link below.

---

## 🚀 How to Run It

### Option A — Google Colab (Recommended)

```python
# Cell 1 — Mount Drive and load models
from google.colab import drive
drive.mount("/content/drive")
import shutil, os
os.makedirs("/content/skin-cancer-detector/models", exist_ok=True)
for f in ["unet_model.tflite", "efficientnet_best.weights.h5", "efficientnet_classifier.keras"]:
    shutil.copy(f"/content/drive/MyDrive/skin-cancer-detector-models/{f}",
                f"/content/skin-cancer-detector/models/{f}")
    print("Loaded:", f)

# Cell 2 — Start Streamlit
import subprocess, time
subprocess.Popen(["streamlit", "run", "/content/skin-cancer-detector/app.py",
                  "--server.port", "8501", "--server.headless", "true"])
time.sleep(8)

# Cell 3 — Get public URL via ngrok
NGROK_TOKEN = "your_token_here"  # from dashboard.ngrok.com
!ngrok authtoken {NGROK_TOKEN}
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN
url = ngrok.connect(8501)
print("App live at:", url)
```

### Option B — Run Locally

```bash
# Clone the repo
git clone https://github.com/Bilal-Butt/Skin-cancer-detection-webapp.git
cd Skin-cancer-detection-webapp

# Install dependencies
pip install -r requirements.txt

# Download models (see Drive link below) and place in models/

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📥 Download Trained Models

Model files are hosted on Google Drive:

👉 **[Download Models — Google Drive](#)** ← *(https://drive.google.com/drive/folders/18guFp51rYuHGRm4j1sXKn2-zqnH1fDu4?usp=drive_link)*

After downloading, place them here:
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

Install everything with:
```bash
pip install -r requirements.txt
```

---

## 💡 Key Technical Decisions

**Why TFLite for the segmentation model?**
The U-Net is converted to TensorFlow Lite — a compressed, optimized format that is significantly faster for inference and smaller in memory, making it ideal for web deployment.

**Why class weights during training?**
HAM10000 is heavily imbalanced — Melanocytic Nevus (NV) makes up roughly 67% of all images. Without class weights, the model just predicts NV for everything and still looks accurate. Class weights force it to genuinely learn the rare but dangerous classes like Melanoma.

**Why skip connections in U-Net?**
Standard encoder-decoder networks lose spatial detail during downsampling. Skip connections pass encoder feature maps directly to the decoder, preserving fine-grained boundary information — critical for accurate lesion edge detection.

**Why EfficientNetB0 over DenseNet201?**
EfficientNetB0 achieves comparable accuracy with dramatically fewer parameters and faster inference time — the right trade-off for a real-time web application.

---

## 🖥️ System Specs Used for Training

| Component | Spec |
|-----------|------|
| Processor | Core i7, 2.30 GHz |
| Training Environment | Google Colab (T4 GPU) |
| Language | Python 3.10 |
| Framework | TensorFlow / Keras |

---

## 🌍 Impact

**Societal Impact:**
- Increases accessibility to early skin cancer screening regardless of location
- Enables self-screening from home before visiting a specialist
- Promotes awareness and preventive behaviour around skin health
- Supports dermatologists by handling initial screening, letting them focus on complex cases

**Healthcare Impact:**
- Reduces unnecessary invasive procedures through accurate early detection
- Lowers overall healthcare costs for patients and systems
- Enables real-time skin lesion analysis via web and mobile platforms

---

## 📚 References

1. WHO — Global skin cancer burden statistics
2. Global Cancer Statistics 2020 — Melanoma in Pakistan
3. Ronneberger O., Fischer P., Brox T. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI
4. Tan M., Le Q. (2019) — EfficientNet: Rethinking Model Scaling for CNNs, Google Brain
5. Tschandl P., Rosendahl C., Kittler H. (2018) — HAM10000 Dataset
6. ISIC Archive — International Skin Imaging Collaboration
7. Aksoy S. (2025) — Skin Lesion Segmentation Using U-Net, IJFMR vol. 7

---

## ⚠️ Disclaimer

> This tool is built for **educational and research purposes only**. It is **not a certified medical device** and should **never replace professional medical advice, diagnosis, or treatment**. Always consult a licensed and qualified dermatologist for any skin health concerns.

---

## 👨‍💻 Author

**Muhammad Bilal Butt**
- 🎓 BS Computer Engineering — COMSATS University Islamabad, Abbottabad Campus
- 💻 GitHub: [@Bilal-Butt](https://github.com/Bilal-Butt)
- 📧 Email: mics.pes@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*If this project helped you or sparked your interest, a ⭐ on GitHub means a lot — it helps others find it too!*
