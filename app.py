import streamlit as st
import tensorflow as tf
import numpy as np
import json
import hashlib
from pathlib import Path
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PlantGuard AI | Plant Health Intelligence",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = Path("models/plantguard_best.keras")
CLASS_PATH = Path("models/class_names.json")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #172b22;
    --muted: #718078;
    --green: #1e714c;
    --green-dark: #103d2b;
    --mint: #eaf5ee;
    --line: #e5ece7;
    --surface: #ffffff;
    --canvas: #f4f7f4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(circle at 8% 2%, rgba(190, 226, 200, .28), transparent 26rem),
        radial-gradient(circle at 95% 12%, rgba(207, 232, 216, .30), transparent 24rem),
        var(--canvas);
}

.block-container {
    max-width: 1320px;
    padding: 2rem 2.2rem 3.5rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(118deg, #102f24 0%, #17563a 52%, #27815a 100%);
    padding: 42px 46px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 28px;
    color: #fff;
    margin: 4px 0 30px;
    box-shadow: 0 20px 55px rgba(17, 67, 43, .16);
}

.hero:after {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    right: -80px;
    top: -170px;
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 50%;
    box-shadow:
        0 0 0 35px rgba(255,255,255,.035),
        0 0 0 75px rgba(255,255,255,.025);
    pointer-events: none;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -1.8px;
    color: white;
    margin: 0 0 10px;
}

.hero p {
    color: #d7eee0;
    font-size: 15px;
    margin: 0;
    line-height: 1.7;
}

.hero .eyebrow {
    display: inline-block;
    border: 1px solid rgba(255,255,255,.2);
    background: rgba(255,255,255,.09);
    color: #e7f6ec;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 18px;
}

/* SECTION HEADINGS */

.section-title {
    color: #1a3d2c;
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.4px;
    margin: 8px 0 16px;
}

/* CARDS */

.upload-card {
    background: rgba(255,255,255,.94);
    padding: 25px;
    border: 1px solid var(--line);
    border-radius: 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 28px rgba(29, 58, 40, .045);
}

.empty-card {
    background: rgba(255,255,255,.94);
    padding: 55px 20px;
    border: 1px solid var(--line);
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 8px 28px rgba(29, 58, 40, .045);
}

.empty-icon {
    font-size: 52px;
}

.empty-card h3 {
    color: #17472e;
    font-family: 'Manrope', sans-serif;
}

.empty-card p {
    color: #718277;
}

/* RESULT DETAILS */

.result-label {
    color: #839087;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.35px;
}

.confidence {
    background: linear-gradient(135deg, #edf8f0, #e5f2e9);
    border: 1px solid #d7eadc;
    padding: 19px 21px;
    border-radius: 17px;
    margin-top: 10px;
}

.confidence h2 {
    color: #17633d;
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    letter-spacing: -1px;
    margin: 2px 0 0;
}

.confidence p {
    color: #668171;
    font-size: 13px;
    font-weight: 600;
    margin: 0;
}

.info-card {
    background: #f0f7f1;
    padding: 17px 19px;
    border-radius: 14px;
    border: 1px solid #dcebe0;
    border-left: 4px solid #27834e;
    color: #315540;
    line-height: 1.65;
    margin-top: 12px;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(135deg, #1d7049, #155638);
    color: white;
    border: 0;
    border-radius: 13px;
    padding: 13px 18px;
    min-height: 50px;
    font-weight: 800;
    box-shadow: 0 8px 18px rgba(24, 102, 64, .16);
    transition: transform .18s ease, box-shadow .18s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #155d3b, #10482f);
    color: white;
    border: 0;
    transform: translateY(-1px);
    box-shadow: 0 11px 22px rgba(24, 102, 64, .22);
}

/* UPLOADER */

div[data-testid="stFileUploader"] {
    background: #f8fbf8;
    border: 1.5px dashed #a9c8b1;
    border-radius: 16px;
    padding: 15px;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #39845a;
}

div[data-testid="stImage"] img {
    border-radius: 17px;
    border: 1px solid #e4ece6;
}

/* PROGRESS BAR */

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2c8b5a, #77bd88);
    border-radius: 999px;
}

.stProgress > div > div {
    background: #eaf0eb;
    border-radius: 999px;
}

.stAlert {
    border-radius: 14px;
}

div[data-testid="stCaptionContainer"] p {
    color: #87938b;
    font-size: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #87948b;
    font-size: 12px;
    padding: 30px 10px 10px;
    border-top: 1px solid #e5ece7;
    margin-top: 28px;
}

/* MOBILE */

@media (max-width: 800px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 30px 24px;
        border-radius: 22px;
    }

    .upload-card {
        padding: 19px;
        border-radius: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class names not found: {CLASS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)

    return model, class_names


try:
    model, class_names = load_model()

except Exception as error:
    st.error(f"Unable to load PlantGuard AI model: {error}")
    st.stop()


# ==========================================
# DISEASE INFORMATION
# ==========================================

disease_info = {

    "Apple___Apple_scab": (
        "Apple Scab",
        "A fungal disease that causes dark, scabby spots on apple leaves and fruit.",
        "Remove infected fallen leaves and improve air circulation. Ask an agricultural expert about suitable fungicide options."
    ),

    "Grape___Black_rot": (
        "Grape Black Rot",
        "A fungal disease that can cause brown leaf lesions and dark, shriveled fruit.",
        "Remove infected plant material and improve vineyard airflow. Seek local agricultural advice."
    ),

    "Potato___Early_blight": (
        "Potato Early Blight",
        "A fungal disease often associated with concentric brown spots on older leaves.",
        "Remove heavily infected leaves, avoid overhead watering and follow local crop protection guidance."
    ),

    "Tomato___healthy": (
        "Healthy Tomato",
        "The model recognizes this image as belonging to the healthy tomato category.",
        "Continue regular watering, monitor plant growth and inspect leaves periodically."
    ),

    "Corn_(maize)___Common_rust_": (
        "Corn Common Rust",
        "A fungal disease that produces rust-colored pustules on corn leaves.",
        "Monitor disease spread and consult local agricultural guidance for management."
    )
}


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((160, 160))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "class": class_names[index],
            "confidence": float(predictions[index]) * 100
        })

    return results


# ==========================================
# FORMAT CLASS NAMES
# ==========================================

def format_name(name):

    plant, disease = name.split("___", 1)

    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()

    return plant, disease


# HERO SECTION
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-powered plant intelligence</div>
    <h1>🌿 PlantGuard AI</h1>
    <p class="hero-subtitle">
        Understand your plant's health, one leaf at a time.
    </p>
    <p class="hero-description">
        Upload a clear leaf photo to explore possible plant conditions
        with our MobileNetV2 deep-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN LAYOUT
# ==========================================

left, right = st.columns([1, 1.15], gap="large")


# ==========================================
# LEFT PANEL - IMAGE UPLOAD
# ==========================================

with left:

    st.markdown(
        '<div class="section-title">📸 Upload Leaf Image</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png"],
            key="leaf_uploader"
        )

        st.caption("Supported formats: JPG, JPEG and PNG")

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            image_bytes = uploaded_file.getvalue()
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            # Clear old result when a new image is uploaded
            if st.session_state.get("image_hash") != image_hash:

                st.session_state.pop("prediction_results", None)
                st.session_state["image_hash"] = image_hash

            st.image(
                image,
                caption="Your uploaded leaf",
                use_container_width=True
            )

            st.success("Image uploaded successfully!")

            analyze = st.button(
                "🔍 Analyze Leaf",
                use_container_width=True,
                type="primary"
            )

            if analyze:

                with st.spinner("AI is analyzing your leaf..."):

                    results = predict_image(image)

                    st.session_state["prediction_results"] = results

        except Exception as error:

            st.error(f"Unable to process this image: {error}")

    else:

        # Clear result when the uploaded image is removed
        st.session_state.pop("prediction_results", None)
        st.session_state.pop("image_hash", None)

        st.markdown("""
        <div class="empty-card">
            <div class="empty-icon">🍃</div>
            <h3>Your plant starts here</h3>
            <p>
                Upload a clear leaf photograph to begin AI analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# RIGHT PANEL - ANALYSIS RESULTS
# ==========================================

with right:

    st.markdown(
        '<div class="section-title">🧪 AI Analysis Results</div>',
        unsafe_allow_html=True
    )

    if "prediction_results" in st.session_state:

        results = st.session_state["prediction_results"]

        best = results[0]

        plant, disease = format_name(best["class"])

        confidence = best["confidence"]

        # RESULT CARD - NATIVE STREAMLIT
        with st.container(border=True):

            st.caption("DETECTED PLANT")

            st.subheader(f"🌿 {plant}")

            st.caption("PREDICTED CONDITION")

            st.subheader(disease)

            st.divider()

            st.markdown("""
            <div class="confidence">
                <p>AI Prediction Confidence</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.progress(min(confidence / 100, 1.0))

        # TOP 3 PREDICTIONS
        st.markdown("#### 📊 Top 3 Predictions")

        for rank, item in enumerate(results, start=1):

            plant_name, disease_name = format_name(item["class"])

            score = item["confidence"]

            if 0 < score < 0.01:
                display_score = "<0.01%"
            else:
                display_score = f"{score:.2f}%"

            st.write(
                f"**{rank}. {plant_name} — {disease_name}**"
            )

            st.caption(f"Confidence: {display_score}")

            st.progress(min(score / 100, 1.0))

        # DISEASE INFORMATION
        st.markdown("#### 🌱 Plant Health Information")

        if best["class"] in disease_info:

            title, description, advice = disease_info[best["class"]]

            with st.container(border=True):

                st.markdown(f"**{title}**")

                st.write(description)

                st.markdown("**Suggested care guidance**")

                st.markdown(
                    f'<div class="info-card">{advice}</div>',
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Specific care information for this category "
                "is not yet available. Consult a local agricultural expert."
            )

        st.caption(
            "Predictions are model estimates, not confirmed diagnoses. "
            "Confidence scores are not guarantees of accuracy."
        )

    else:

       st.markdown("""
<div class="empty-card" style="padding:65px 15px;">
    <div class="empty-icon">🔬</div>
    <h3>Waiting for analysis</h3>
    <p>Your AI prediction and plant health information
    will appear here after analysis.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🌿 PlantGuard AI &nbsp; | &nbsp;
    Powered by TensorFlow and MobileNetV2
    <br><br>
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)
