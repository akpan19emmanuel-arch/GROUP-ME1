import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import keras

# ── OOD detection (shared utility) ───────────────────────────────────────────
from ood import load_class_stats, build_feature_extractor, is_ood

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mango Formalin Detector",
    page_icon="🥭",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = Path(__file__).parent / "model" / "mango_rotten_formalin_model.keras"
CLASS_STATS_PATH = Path(__file__).parent / "model" / "class_stats.npz"
# Alphabetical: Formalin_Mixed (index 0), Rotten (index 1)
# raw_prob = P(Rotten)
CLASS_NAMES     = ["Formalin_Mixed", "Rotten"]
IMAGE_SIZE      = (224, 224)
THRESHOLD       = 0.5

# ── Model + OOD resources (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model          = keras.models.load_model(MODEL_PATH)
    feat_extractor = build_feature_extractor(model)
    stats          = load_class_stats(CLASS_STATS_PATH)
    return model, feat_extractor, stats

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(model, pil_image: Image.Image):
    """raw_prob = P(Rotten) — Rotten is CLASS_NAMES[1] (alphabetically second)."""
    img   = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr   = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[int(prob >= THRESHOLD)]
    conf  = prob if prob >= THRESHOLD else 1.0 - prob
    return label, conf, prob

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🥭 Mango Formalin Detector")
st.caption(
    "Upload a mango image and the model will classify it as "
    "**Rotten** or **Formalin-Mixed** using a MobileNetV2 transfer-learning model."
)
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** MobileNetV2 (fine-tuned)  
        **Classes:** Formalin Mixed · Rotten  
        **Input size:** 224 × 224 px  
        **Dataset:** [Fruits Disease Dataset](https://www.kaggle.com/datasets/saravanansri/fruits-disease-dataset) — Kaggle  

        ---
        > ⚠️ **Disclaimer:** This tool is for educational and food-safety research
        > purposes only. It is **not** a certified food inspection system.
        """
    )
    st.divider()
    st.markdown("**Model file:**")
    st.code(str(MODEL_PATH.name), language=None)
    ood_stats_ok = CLASS_STATS_PATH.exists()
    st.markdown(f"**OOD stats:** {'✅ loaded' if ood_stats_ok else '⚠️ not found (Layer 2 disabled)'}")

uploaded = st.file_uploader(
    "Upload a mango image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Clear, close-up images of the mango surface produce the best results.",
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        model, feat_extractor, stats = load_resources()

        with st.spinner("Analysing…"):
            label, confidence, raw_prob = predict(model, image)
            rejected, reason = is_ood(feat_extractor, stats, image, raw_prob)

        if rejected:
            st.warning(f"⚠️ **Image not recognised**\n\n{reason}", icon="🚫")
        else:
            color = "#e74c3c" if label == "Rotten" else "#e67e22"
            st.markdown(
                f"""
                <div style="background-color:{color}22;border-left:5px solid {color};
                padding:16px 20px;border-radius:6px;margin-bottom:12px;">
                <span style="font-size:2rem;font-weight:700;color:{color};">
                {label.replace("_", " ")}</span></div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

            with st.expander("Raw probabilities"):
                rotten_prob   = raw_prob
                formalin_prob = 1.0 - raw_prob
                st.write(f"**Rotten:** {rotten_prob * 100:.2f}%")
                st.progress(rotten_prob)
                st.write(f"**Formalin Mixed:** {formalin_prob * 100:.2f}%")
                st.progress(formalin_prob)

    st.divider()
    st.caption("⚠️ For educational purposes only. Do not use as the sole basis for food safety decisions.")

else:
    st.info("👆 Upload a mango image above to get a prediction.", icon="🖼️")
    with st.expander("See sample images from the dataset"):
        sample_path = Path(__file__).parent / "results" / "sample_images.png"
        if sample_path.exists():
            st.image(str(sample_path), caption="Training data samples", use_container_width=True)
        else:
            st.write("Sample image not available.")
