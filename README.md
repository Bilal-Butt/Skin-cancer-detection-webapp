
# 🔬 Skin Cancer Detection Web App

> *Early detection saves lives. This project is a step toward making dermatological AI accessible to everyone.*

---

## 👋 What Is This Project?

Hey! Welcome to my Skin Cancer Detection Web App — a project I built to explore how deep learning can assist in identifying skin lesions from dermoscopy images.

The idea is simple: you upload a photo of a skin lesion, and the app does two things automatically:

1. **Draws a mask** around the suspicious area (segmentation)
2. **Identifies what type** of lesion it is and flags the risk level (classification)

It is not a replacement for a real dermatologist — not even close. But it is a pretty cool demonstration of how AI can be a powerful first-pass screening tool in medical imaging.

---

## 🎯 Why I Built This

Skin cancer is one of the most common cancers worldwide, and the good news is — when caught early, it is also one of the most treatable. The challenge is that access to dermatologists is not equal everywhere in the world.

I wanted to build something that shows how machine learning can bridge that gap, even a little. This project taught me a ton about medical image segmentation, transfer learning, model deployment, and building real-world AI pipelines from scratch.

---

## ✨ What It Can Do

| Feature | Description |
|--------|-------------|
| 🎯 **Lesion Segmentation** | Highlights exactly where the lesion is using a U-Net model |
| 🏷️ **Lesion Classification** | Identifies which of 7 types the lesion belongs to |
| ⚠️ **Risk Assessment** | Flags lesions as High / Moderate / Low risk |
| 📊 **Confidence Scores** | Shows how confident the model is across all 7 classes |
| 🖼️ **Visual Overlay** | Displays original image, mask, and overlay side by side |

---

## 🧠 The Models Behind It

### Model 1 — U-Net (Segmentation)
The U-Net is a classic architecture for medical image segmentation. I built a lightweight version that:
- Takes a 256x256 dermoscopy image as input
- Outputs a binary mask (white = lesion, black = background)
- Uses skip connections to preserve fine spatial details
- Trained from scratch on the HAM10000 dataset

### Model 2 — EfficientNetB0 (Classification)
Instead of training a classifier from zero, I used transfer learning — taking EfficientNetB0 pretrained on ImageNet and fine-tuning it on our skin lesion data.

---

## 📂 The 7 Skin Lesion Classes

| Abbreviation | Full Name | Risk Level |
|-------------|-----------|------------|
| AKIEC | Actinic Keratosis | 🟡 Moderate |
| BCC | Basal Cell Carcinoma | 🔴 High |
| BKL | Benign Keratosis | 🟢 Low |
| DF | Dermatofibroma | 🟢 Low |
| MEL | Melanoma | 🔴 High |
| NV | Melanocytic Nevus | 🟢 Low |
| VASC | Vascular Lesion | 🟢 Low |

---

## 🗂️ Project Structure

\`\`\`
skin-cancer-detection-webapp/
├── app.py                  
├── segmentation.py         
├── classification.py       
├── finetune.py             
├── train_unet.py           
├── requirements.txt        
├── README.md               
└── models/
    ├── unet_model.tflite
    ├── efficientnet_best.weights.h5
    └── efficientnet_classifier.keras
\`\`\`

---

## 🚀 How to Run It Locally

\`\`\`bash
git clone https://github.com/Bilal-Butt/Skin-cancer-detection-webapp.git
cd Skin-cancer-detection-webapp
pip install -r requirements.txt
streamlit run app.py
\`\`\`

---

## 📦 Dependencies

\`\`\`
streamlit
tensorflow >= 2.10
numpy
pandas
pillow
matplotlib
scikit-learn
\`\`\`

---

## ⚠️ Disclaimer

This tool is built for educational and research purposes only. It is not a medical device and should never be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for any skin concerns.

---

## 👨‍💻 Author

**Bilal Butt**
- GitHub: [@Bilal-Butt](https://github.com/Bilal-Butt)
- Email: mics.pes@gmail.com

---

*If you found this project interesting or helpful, consider giving it a ⭐ — it genuinely means a lot!*
