import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from datetime import date

# classification.py no longer sets a precision policy at import time, so the
# default float32 is used everywhere. Inference is safe on CPU and GPU alike.
from segmentation import run_segmentation, load_segmentation_model
from classification import build_classifier, run_classification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan AI",
    page_icon="🌸",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #fff0f6 0%, #f0f4ff 50%, #f0fff8 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(120deg, #ff6eb4, #ff9a5c, #ffcd3c, #6be5a0, #5cb8ff);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    border-radius: 24px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(255,110,180,0.25);
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: white;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
    margin: 0;
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 600;
}

.card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
    border: 1.5px solid rgba(255,255,255,0.8);
}
.card-title {
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 1rem;
    color: #333;
}

.result-card {
    border-radius: 18px;
    padding: 1.2rem 1rem;
    text-align: center;
    font-family: 'Nunito', sans-serif;
}
.result-card .label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 6px;
}
.result-card .value {
    font-size: 1.25rem;
    font-weight: 900;
    line-height: 1.3;
}
.card-pink   { background: linear-gradient(135deg, #fff0f7, #ffe4f2); border: 2px solid #ffcce8; }
.card-yellow { background: linear-gradient(135deg, #fffdf0, #fff5cc); border: 2px solid #ffe78a; }
.card-blue   { background: linear-gradient(135deg, #f0f6ff, #deeeff); border: 2px solid #b8d8ff; }

.risk-badge { display:inline-block; padding:5px 16px; border-radius:50px; font-weight:800; font-size:0.95rem; }
.risk-low      { background:#d6f5e8; color:#1a7a4a; }
.risk-moderate { background:#fff5cc; color:#8a6000; }
.risk-high     { background:#ffdede; color:#c0392b; }

.img-caption {
    text-align: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #999;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.disclaimer {
    background: linear-gradient(135deg, #fff8e1, #fff3cd);
    border: 2px solid #ffe082;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #7a5c00;
    font-weight: 600;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-top: 1rem;
}

.empty-state {
    background: white;
    border-radius: 20px;
    padding: 3.5rem 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 2.5px dashed #ffcce8;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌸 SkinScan AI</h1>
    <p>AI-powered skin lesion segmentation &amp; classification &nbsp;·&nbsp; HAM10000 &nbsp;·&nbsp; EfficientNetB3</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    Load and cache both models once at startup. Reused across every upload —
    no redundant disk reads per inference call.
    """
    base         = os.path.dirname(os.path.abspath(__file__))
    unet_path    = os.path.join(base, "models", "unet_model.tflite")
    weights_path = os.path.join(base, "models", "efficientnet_best.weights.h5")

    for path, name in [(unet_path, "unet_model.tflite"),
                       (weights_path, "efficientnet_best.weights.h5")]:
        if not os.path.isfile(path):
            st.error(
                f"❌ Model file not found: **{name}**\n\n"
                "Place trained model files in the `models/` folder next to "
                "`app.py`. Run `train_unet.py` and `finetune.py` to generate them."
            )
            st.stop()

    interpreter   = load_segmentation_model(unet_path)
    classifier, _ = build_classifier()
    classifier.load_weights(weights_path)
    classifier.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return interpreter, classifier


with st.spinner("Loading AI models..."):
    interpreter, classifier = load_models()

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.65], gap="large")

# ── LEFT: Patient info + Upload ───────────────────────────────────────────────
with left_col:

    st.markdown('<div class="card"><div class="card-title">👤 Patient Information</div>',
                unsafe_allow_html=True)

    patient_name = st.text_input("Full Name", placeholder="e.g. Jane Smith")

    col_age, col_sex = st.columns(2)
    with col_age:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    with col_sex:
        patient_sex = st.selectbox("Sex", ["Select", "Male", "Female", "Other"])

    col_date, col_loc = st.columns(2)
    with col_date:
        exam_date = st.date_input("Exam Date", value=date.today())
    with col_loc:
        lesion_location = st.selectbox("Lesion Location", [
            "Select", "Face", "Scalp", "Neck", "Chest", "Back",
            "Abdomen", "Upper arm", "Forearm", "Hand",
            "Thigh", "Lower leg", "Foot", "Other"
        ])

    patient_notes = st.text_area("Clinical Notes",
                                 placeholder="Any relevant observations...",
                                 height=90)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🖼️ Upload Dermoscopy Image</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with right_col:

    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:4rem;margin-bottom:1rem;">🔬</div>
            <div style="font-size:1.2rem;font-weight:800;color:#cc3d88;margin-bottom:0.5rem;">
                Ready for analysis
            </div>
            <div style="color:#bbb;font-size:0.95rem;font-weight:600;line-height:1.6;">
                Fill in the patient details on the left<br>
                and upload a skin lesion image<br>
                to get your AI-powered results
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not open image: {e}")
            st.stop()

        # ── Step 1: Segmentation — pass the cached interpreter ────────────────
        with st.spinner("Running AI segmentation..."):
            try:
                mask = run_segmentation(image, interpreter=interpreter)
            except Exception as e:
                st.error(f"❌ Segmentation failed: {e}")
                st.stop()

        # ── Step 2: Classification ────────────────────────────────────────────
        with st.spinner("Classifying lesion..."):
            try:
                results = run_classification(image, classifier, mask=mask)
            except Exception as e:
                st.error(f"❌ Classification failed: {e}")
                st.stop()

        predicted_class = results["predicted_class"]
        risk_level      = results["risk_level"]
        confidence      = results["confidence"]

        # ── Visuals ───────────────────────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">🖼️ Analysis Visuals</div>',
                    unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)

        with ic1:
            st.image(image.resize((256, 256)), use_container_width=True)
            st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)

        with ic2:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            st.image(mask_img, use_container_width=True)
            st.markdown('<div class="img-caption">Segmentation Mask</div>',
                        unsafe_allow_html=True)

        with ic3:
            fig, ax = plt.subplots(figsize=(3, 3))
            fig.patch.set_facecolor("none")
            ax.imshow(np.array(image.resize((256, 256))))
            ax.imshow(mask, cmap="RdPu", alpha=0.55)
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('<div class="img-caption">Overlay</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Diagnosis summary ─────────────────────────────────────────────────
        risk_css   = {"Low": "risk-low", "Moderate": "risk-moderate",
                      "High": "risk-high"}.get(risk_level, "risk-low")
        risk_emoji = {"Low": "🟢", "Moderate": "🟡",
                      "High": "🔴"}.get(risk_level, "⚪")

        st.markdown('<div class="card"><div class="card-title">📊 Diagnosis Summary</div>',
                    unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown(f"""
            <div class="result-card card-pink">
                <div class="label">Predicted Class</div>
                <div class="value" style="color:#c0396b;font-size:1rem;">{predicted_class}</div>
            </div>""", unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="result-card card-yellow">
                <div class="label">Risk Level</div>
                <div class="value">
                    <span class="risk-badge {risk_css}">{risk_emoji} {risk_level}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with mc3:
            st.markdown(f"""
            <div class="result-card card-blue">
                <div class="label">Confidence</div>
                <div class="value" style="color:#1a6fcc;">{confidence:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Patient record summary ─────────────────────────────────────────────
        if patient_name:
            st.markdown('<div class="card"><div class="card-title">🗒️ Patient Record</div>',
                        unsafe_allow_html=True)
            pr1, pr2, pr3 = st.columns(3)
            pr1.markdown(f"**Name**<br>{patient_name}", unsafe_allow_html=True)
            pr2.markdown(f"**Age / Sex**<br>{patient_age} / {patient_sex}",
                         unsafe_allow_html=True)
            pr3.markdown(f"**Exam Date**<br>{exam_date.strftime('%d %b %Y')}",
                         unsafe_allow_html=True)
            if lesion_location != "Select":
                st.markdown(f"**Lesion Location:** {lesion_location}")
            if patient_notes:
                st.markdown(f"**Notes:** {patient_notes}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Confidence chart ──────────────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">📈 All Class Scores</div>',
                    unsafe_allow_html=True)

        classes = list(results["all_scores"].keys())
        scores  = list(results["all_scores"].values())
        colors  = ["#ff6eb4" if c == predicted_class else "#c8e6ff" for c in classes]

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        fig2.patch.set_facecolor("none")
        ax2.set_facecolor("none")
        bars = ax2.barh(classes, scores, color=colors, height=0.55, edgecolor="none")
        for bar, score in zip(bars, scores):
            ax2.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{score:.1%}", va="center", fontsize=9,
                     fontweight="bold", color="#666")
        ax2.set_xlim(0, 1.18)
        ax2.set_xlabel("Confidence", fontsize=9, color="#aaa")
        ax2.axvline(x=0.5, color="#eee", linestyle="--", linewidth=1.2)
        ax2.tick_params(colors="#777", labelsize=9)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            <span style="font-size:1.3rem;flex-shrink:0;">⚠️</span>
            <span>This tool is for <strong>educational purposes only</strong> and is not a
            substitute for professional medical diagnosis.
            Always consult a licensed dermatologist.</span>
        </div>
        """, unsafe_allow_html=True)
