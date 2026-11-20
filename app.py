import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from google import genai

# 1. Page Configuration (Wide layout for modern dashboard feel)
st.set_page_config(
    page_title="AgriVision AI | Smart Plant Care",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom Botanical UI/UX Styling
st.markdown("""
    <style>
    /* Main Background & Font */
    .stApp {
        background-color: #F4F7F2;
        color: #1E2D24;
    }
    
    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #2D4F3E 0%, #4A7C59 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(45, 79, 62, 0.15);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #DCE8E0;
        margin-top: 0.4rem;
        font-size: 1.05rem;
    }

    /* Card Containers */
    .status-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 14px;
        border-left: 6px solid #4A7C59;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .status-card.thirsty {
        border-left-color: #D9534F;
        background-color: #FFF9F9;
    }
    
    /* Primary Action Buttons */
    .stButton > button {
        background-color: #2D4F3E !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #3E6B54 !important;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Hero Header
st.markdown("""
    <div class="hero-banner">
        <h1>🌿 AgriVision AI</h1>
        <p>Instant Hydration Detection & AI Botanist Diagnostics</p>
    </div>
""", unsafe_allow_html=True)

# Optional Sidebar for API Key fallback (keeps main UI clean)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input(
        "Gemini API Key (Optional if set in env)", 
        type="password",
        help="Paste your Gemini API key here if not configured in environment variables."
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
            # Clean Teachable Machine index prefix (e.g., "0 Thirsty" -> "Thirsty")
            return [line.strip().split(" ", 1)[-1] for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error("⚠️ Missing `labels.txt` file in your project folder!")
        return []

model = load_my_model()
class_names = load_labels()

# 5. Two-Column Dashboard Layout
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("1. Capture or Upload Plant")
    tab_upload, tab_cam = st.tabs(["📁 Upload Photo", "📸 Live Camera"])
    
    img_file = None
    with tab_upload:
        upload_img = st.file_uploader("Select a clear leaf or soil photo", type=["jpg", "png", "jpeg"])
        if upload_img:
            img_file = upload_img

    with tab_cam:
        camera_img = st.camera_input("Snap a picture")
        if camera_img:
            img_file = camera_img

    # Open image ONCE and keep it in memory to prevent file-pointer bugs
    original_image = None
    if img_file is not None:
        original_image = Image.open(img_file).convert("RGB")
        st.image(original_image, caption="Selected Specimen", use_container_width=True)

with col_right:
    st.subheader("2. Diagnostics & Care Report")
    
    if original_image is not None and model is not None and class_names:
        # --- PHASE A: Instant Local Keras Prediction ---
        size = (224, 224)
        processed_img = ImageOps.fit(original_image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(processed_img, dtype=np.float32)
        normalized_img_array = (img_array / 127.5) - 1.0
        data = np.expand_dims(normalized_img_array, axis=0)  # Shape: (1, 224, 224, 3)

        prediction = model.predict(data, verbose=0)
        index = int(np.argmax(prediction))
        class_name = class_names[index]
        confidence = float(prediction[0][index])

        is_thirsty = "thirsty" in class_name.lower()
        card_class = "status-card thirsty" if is_thirsty else "status-card"
        status_icon = "💧" if is_thirsty else "✅"
        advice_text = (
            "Soil/leaves show dehydration signs. Water thoroughly today."
            if is_thirsty else
            "Hydration looks balanced! Keep up your current routine."
        )

        st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin:0;">{status_icon} Hydration Status: {class_name}</h3>
                <p style="margin:0.5rem 0 0 0; color:#4A5568;">{advice_text}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"Model Confidence: **{confidence * 100:.1f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))

        st.divider()

        # --- PHASE B: Deep GenAI Botanist Report ---
        st.markdown("#### 🩺 Deep Health, Disease & Fertilizer Analysis")
        if st.button("✨ Generate Full AI Botanist Report"):
            if not os.environ.get("GEMINI_API_KEY"):
                st.warning("🔑 Please enter your Gemini API Key in the left sidebar (click `>` at top-left) or set `GEMINI_API_KEY` in your environment.")
            else:
                with st.spinner("Inspecting leaves, stems, and soil health..."):
                    try:
                        client = genai.Client()
                        prompt = """
                        Analyze this plant photo carefully. Provide a concise, practical, 
                        gardening-focused care guide using these exact Markdown sections:
                        
                        ### 🌱 Plant Identity
                        Identify the common name and likely variety.
                        
                        ### 🔬 Health & Disease Check
                        Examine leaves, stems, and visible soil for:
                        - Fungal or bacterial infections
                        - Pest infestations
                        - Nutrient deficiencies
                        If healthy, confirm it. If issues exist, give clear step-by-step treatment.
                        
                        ### 💧 Optimal Watering Schedule
                        Specify watering frequency, amount, and soil-check tips for this species.
                        
                        ### 🌿 Manure & Fertilizer Guide
                        Recommend the best organic manure or fertilizer, dosage, and how often to apply it.
                        """
                        
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[original_image, prompt]
                        )
                        
                        with st.container(border=True):
                            st.markdown(response.text)
                            
                    except Exception as e:
                        st.error(f"AI Analysis Error: {e}")
    else:
        st.info("👈 Upload or capture a plant photo on the left to see instant hydration results and unlock AI care diagnostics.")
