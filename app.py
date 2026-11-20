import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from google import genai

# 1. Page Configuration
st.set_page_config(
    page_title="AgriVision AI | Dark Cottagecore Botanist",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Dark Cottagecore + Glassmorphism + Botanical Glow CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Deep Woodland Night Canvas + Ambient Moss Radial Gradients */
    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(76, 124, 89, 0.24) 0%, transparent 38%),
            radial-gradient(circle at 88% 12%, rgba(163, 184, 153, 0.14) 0%, transparent 36%),
            radial-gradient(circle at 75% 85%, rgba(62, 107, 84, 0.22) 0%, transparent 42%),
            linear-gradient(160deg, #0B120E 0%, #111C15 48%, #0A100C 100%);
        color: #E7EFE9;
        font-family: 'Plus Jakarta Sans', sans-serif;
        overflow-x: hidden;
    }

    /* Hide default Streamlit header bar clutter while keeping sidebar toggle */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Floating Botanical Spores / Fireflies Layer */
    .firefly-field {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .spore {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(196, 224, 184, 0.85) 0%, rgba(118, 168, 122, 0.15) 65%, transparent 100%);
        box-shadow: 0 0 12px rgba(168, 213, 162, 0.55);
        animation: drift linear infinite;
    }
    .spore.s1 { width: 7px; height: 7px; left: 8%; top: 82%; animation-duration: 14s; animation-delay: 0s; }
    .spore.s2 { width: 5px; height: 5px; left: 22%; top: 90%; animation-duration: 18s; animation-delay: -4s; }
    .spore.s3 { width: 9px; height: 9px; left: 45%; top: 88%; animation-duration: 21s; animation-delay: -7s; }
    .spore.s4 { width: 6px; height: 6px; left: 68%; top: 92%; animation-duration: 16s; animation-delay: -2s; }
    .spore.s5 { width: 8px; height: 8px; left: 85%; top: 86%; animation-duration: 19s; animation-delay: -9s; }
    .spore.s6 { width: 5px; height: 5px; left: 93%; top: 75%; animation-duration: 15s; animation-delay: -5s; }

    @keyframes drift {
        0% {
            transform: translateY(0px) translateX(0px) scale(0.8);
            opacity: 0;
        }
        15% {
            opacity: 0.85;
        }
        50% {
            transform: translateY(-45vh) translateX(18px) scale(1.1);
            opacity: 0.95;
        }
        85% {
            opacity: 0.6;
        }
        100% {
            transform: translateY(-95vh) translateX(-14px) scale(0.75);
            opacity: 0;
        }
    }

    /* Cottagecore Serif Display Headings */
    h1, h2, h3, .cottage-serif {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        letter-spacing: 0.4px;
    }

    /* Enchanted Greenhouse Hero Banner */
    .cottage-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(31, 56, 42, 0.72) 0%, rgba(18, 33, 24, 0.82) 55%, rgba(56, 86, 63, 0.45) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(169, 204, 162, 0.28);
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow:
            0 20px 45px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(226, 242, 222, 0.22);
        overflow: hidden;
    }
    .cottage-hero::before {
        content: "🌿  ✦  🍃";
        position: absolute;
        right: 28px;
        top: 20px;
        font-size: 1.5rem;
        opacity: 0.45;
        letter-spacing: 8px;
    }
    .cottage-badge {
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        background: rgba(138, 184, 138, 0.16);
        border: 1px solid rgba(169, 204, 162, 0.35);
        color: #CBE3C4;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .cottage-hero h1 {
        color: #F2F8F1 !important;
        font-size: 2.8rem !important;
        line-height: 1.1 !important;
        margin: 0 !important;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .cottage-hero p {
        color: #BDD0BE;
        font-size: 1.05rem;
        margin: 0.65rem 0 0 0;
        max-width: 680px;
    }

    /* Glassmorphism Section Shells */
    .glass-panel {
        background: rgba(20, 34, 26, 0.58);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(164, 199, 158, 0.2);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.32);
        margin-bottom: 1.2rem;
    }

    /* Hydration Status Glass Card */
    .specimen-card {
        background: linear-gradient(145deg, rgba(44, 79, 59, 0.45), rgba(16, 28, 21, 0.78));
        backdrop-filter: blur(14px);
        border: 1px solid rgba(152, 204, 152, 0.32);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    .specimen-card.thirsty {
        background: linear-gradient(145deg, rgba(110, 44, 42, 0.42), rgba(24, 15, 16, 0.82));
        border: 1px solid rgba(235, 125, 118, 0.38);
    }
    .specimen-title {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.9rem !important;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        color: #F4FAF3;
    }
    .specimen-sub {
        color: #C8D8C9;
        margin: 0;
        font-size: 0.96rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 23, 17, 0.65);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(164, 199, 158, 0.16);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #B4C7B6;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(78, 126, 94, 0.65), rgba(42, 74, 53, 0.85)) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(183, 219, 176, 0.35) !important;
    }

    /* File Uploader & Camera Glass Polish */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 27, 20, 0.65) !important;
        border: 1.5px dashed rgba(162, 199, 156, 0.38) !important;
        border-radius: 16px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(198, 232, 191, 0.75) !important;
        background: rgba(25, 44, 33, 0.75) !important;
    }

    /* Glowing Botanical CTA Button */
    .stButton > button {
        background: linear-gradient(135deg, #3D6B4F 0%, #274734 100%) !important;
        color: #F4FAF2 !important;
        border: 1px solid rgba(177, 214, 170, 0.4) !important;
        border-radius: 14px !important;
        padding: 0.72rem 1.3rem !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 8px 22px rgba(24, 48, 34, 0.55), inset 0 1px 0 rgba(255,255,255,0.18) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4E8563 0%, #325B43 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(78, 133, 99, 0.38), inset 0 1px 0 rgba(255,255,255,0.28) !important;
    }

    /* Image Frame */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid rgba(173, 209, 166, 0.28);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }

    /* Progress Bar Tint */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #5F9670 0%, #B8DFB0 100%) !important;
    }
    </style>

    <!-- Floating Botanical Spores / Fireflies -->
    <div class="firefly-field">
        <div class="spore s1"></div>
        <div class="spore s2"></div>
        <div class="spore s3"></div>
        <div class="spore s4"></div>
        <div class="spore s5"></div>
        <div class="spore s6"></div>
    </div>
""", unsafe_allow_html=True)

# 3. Hero Header
st.markdown("""
    <div class="cottage-hero">
        <div class="cottage-badge">✦ Midnight Conservatory &amp; Flora Lab ✦</div>
        <h1>AgriVision AI</h1>
        <p>Whispering-leaf hydration telemetry, botanical disease diagnostics, and tailored nourishment notes crafted for quiet greenhouses.</p>
    </div>
""", unsafe_allow_html=True)

# Optional Sidebar for API Key fallback
with st.sidebar:
    st.markdown("###  Conservatory Key")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        help="Optional if GEMINI_API_KEY is already set in your environment."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

# 4. Load Keras Model & Labels Safely
@st.cache_resource
def load_my_model():
    try:
        return load_model("keras_model.h5", compile=False)
    except Exception as e:
        st.error(f"⚠️ Could not load `keras_model.h5`: {e}")
        return None

@st.cache_data
def load_labels():
    try:
        with open("labels.txt", "r") as f:
            return [line.strip().split(" ", 1)[-1] for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error("⚠️ Missing `labels.txt` file in your project folder!")
        return []

model = load_my_model()
class_names = load_labels()

# 5. Two-Column Glass Dashboard
col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🍂 Specimen Capture")
    tab_upload, tab_cam = st.tabs(["📁 Press / Upload Leaf", "📷 Lantern Camera"])

    img_file = None
    with tab_upload:
        upload_img = st.file_uploader("Choose a plant photograph", type=["jpg", "png", "jpeg"])
        if upload_img:
            img_file = upload_img

    with tab_cam:
        camera_img = st.camera_input("Capture your specimen")
        if camera_img:
            img_file = camera_img

    original_image = None
    if img_file is not None:
        original_image = Image.open(img_file).convert("RGB")
        st.image(original_image, caption="✦ Specimen in Observation", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("###  Botanical Telemetry & Diagnostics")

    if original_image is not None and model is not None and class_names:
        # Phase A: Keras Hydration Check
        size = (224, 224)
        processed_img = ImageOps.fit(original_image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(processed_img, dtype=np.float32)
        normalized_img_array = (img_array / 127.5) - 1.0
        data = np.expand_dims(normalized_img_array, axis=0)

        prediction = model.predict(data, verbose=0)
        index = int(np.argmax(prediction))
        class_name = class_names[index]
        confidence = float(prediction[0][index])

        is_thirsty = "thirsty" in class_name.lower()
        card_cls = "specimen-card thirsty" if is_thirsty else "specimen-card"
        glyph = "🥀" if is_thirsty else "🌿"
        note = (
            "Leaves & soil signal thirst — offer a gentle, thorough drink soon."
            if is_thirsty else
            "Hydration is serene and balanced — maintain your current care rhythm."
        )

        st.markdown(f"""
            <div class="{card_cls}">
                <div class="specimen-title">{glyph} {class_name}</div>
                <p class="specimen-sub">{note}</p>
            </div>
        """, unsafe_allow_html=True)

        st.caption(f"Signal Certainty: **{confidence * 100:.1f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))

        st.markdown("---")

        # Phase B: AI Botanist Deep Diagnostics
        st.markdown("### Apothecary & Care Almanac")
        if st.button(" Unveil Full Botanist Diagnosis"):
            if not os.environ.get("GEMINI_API_KEY"):
                st.warning(" Add your `GEMINI_API_KEY` in the left sidebar or environment variables first.")
            else:
                with st.spinner("Reading leaf veins, soil whispers, and care lore..."):
                    try:
                        client = genai.Client()
                        prompt = """
                        Analyze this plant photo carefully. Provide a concise, practical,
                        gardening-focused care guide using these exact Markdown sections:

                        ###  Plant Identity
                        Identify the common name and likely variety.

                        ###  Health & Disease Check
                        Examine leaves, stems, and visible soil for:
                        - Fungal or bacterial infections
                        - Pest infestations
                        - Nutrient deficiencies
                        If healthy, confirm it. If issues exist, give clear step-by-step treatment.

                        ###  Optimal Watering Schedule
                        Specify watering frequency, amount, and soil-check tips for this species.

                        ###  Manure & Fertilizer Guide
                        Recommend the best organic manure or fertilizer, dosage, and how often to apply it.
                        """
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[original_image, prompt]
                        )
                        with st.container(border=True):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Botanist lookup failed: {e}")
    else:
        st.markdown("""
            <div class="specimen-card">
                <div class="specimen-title"> Awaiting a Specimen</div>
                <p class="specimen-sub">Upload a leaf photo or snap one with your camera on the left to begin hydration telemetry and deep plant diagnostics.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
