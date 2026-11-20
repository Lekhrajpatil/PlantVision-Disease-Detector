import os
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — explicit colors so text is visible in BOTH light & dark mode ─────────
st.markdown("""
<style>
/* ===== GLOBAL RESET for dark-mode safety ===== */
.pd-card, .pd-info, .pd-warn, .pd-danger,
.pd-result-title, .pd-label, .pd-caption,
.pd-stat-box, .pd-about-box, .pd-low-conf {
    color: #1a1a1a !important;
}

/* ===== BASE CARD ===== */
.pd-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.10);
    margin: 10px 0 16px 0;
    border: 1px solid #e0e0e0;
}

/* ===== RESULT HEADER INSIDE CARD ===== */
.pd-plant-name {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1b5e20 !important;
    margin-bottom: 2px;
}
.pd-disease-name {
    font-size: 1.25rem;
    font-weight: 700;
    color: #b71c1c !important;
    margin-bottom: 10px;
}
.pd-meta {
    font-size: 0.95rem;
    color: #333333 !important;
    margin-top: 4px;
}

/* ===== COLOURED INFO BOXES ===== */
.pd-info {
    background: #f1f8e9;
    border-left: 5px solid #558b2f;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1a1a1a !important;
    font-size: 0.95rem;
    line-height: 1.6;
}
.pd-warn {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1a1a1a !important;
    font-size: 0.95rem;
}
.pd-danger {
    background: #ffebee;
    border-left: 5px solid #c62828;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1a1a1a !important;
    font-size: 0.95rem;
}
.pd-low-conf {
    background: #fff3e0;
    border-left: 5px solid #ef6c00;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1a1a1a !important;
    font-size: 0.95rem;
}

/* ===== CONFIDENCE BAR LABEL ===== */
.pd-bar-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #222222 !important;
    background: #f5f5f5;
    padding: 5px 10px;
    border-radius: 6px;
    margin-bottom: 2px;
    display: block;
}
.pd-bar-label-active {
    background: #e8f5e9;
    color: #1b5e20 !important;
}

/* ===== STAT BOXES ===== */
.pd-stat-box {
    background: #ffffff;
    border: 2px solid #c8e6c9;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    color: #1a1a1a !important;
}
.pd-stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: #2e7d32 !important;
}
.pd-stat-label {
    font-size: 0.85rem;
    color: #444444 !important;
    margin-top: 2px;
}

/* ===== HISTORY TABLE ROW ===== */
.pd-hist-row {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #1a1a1a !important;
    font-size: 0.9rem;
}

/* ===== ABOUT BOX ===== */
.pd-about-box {
    background: #ffffff;
    border: 1px solid #dcedc8;
    border-radius: 12px;
    padding: 20px 24px;
    color: #1a1a1a !important;
    font-size: 0.95rem;
    line-height: 1.7;
}
.pd-about-box h4 {
    color: #1b5e20 !important;
    margin-bottom: 8px;
}

/* ===== SUPPORTED DISEASE BADGE in sidebar ===== */
.pd-disease-badge {
    background: #e8f5e9;
    border-radius: 8px;
    padding: 7px 12px;
    margin: 5px 0;
    color: #1b5e20 !important;
    font-size: 0.88rem;
    font-weight: 600;
    display: block;
}

/* ===== UPLOAD HINT ===== */
.pd-upload-hint {
    text-align: center;
    color: #555555 !important;
    font-size: 0.85rem;
    margin-top: -6px;
    background: transparent;
}

/* ===== EMPTY STATE ===== */
.pd-empty {
    background: #f9fbe7;
    border: 2px dashed #aed581;
    border-radius: 14px;
    padding: 50px 24px;
    text-align: center;
    color: #444444 !important;
}
.pd-empty-sub {
    font-size: 0.9rem;
    color: #666666 !important;
    margin-top: 6px;
}

/* ===== SECTION HEADINGS ===== */
.pd-section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1b5e20 !important;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 2px solid #c8e6c9;
}

/* ===== BULLET ITEMS ===== */
.pd-bullet {
    background: #f9fbe7;
    border-radius: 6px;
    padding: 7px 12px;
    margin: 4px 0;
    color: #1a1a1a !important;
    font-size: 0.92rem;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_disease.h5")

@st.cache_resource(show_spinner="⏳ Loading model…")
def load_disease_model():
    return load_model(MODEL_PATH)

model = load_disease_model()

# ── Full disease knowledge base (3 classes the model supports) ─────────────────
DISEASE_INFO = {
    "Corn-Common_rust": {
        "plant":   "Corn (Maize)",
        "disease": "Common Rust",
        "emoji":   "🌽",
        "severity": "Moderate",
        "sev_class": "pd-warn",
        "action_urgency": "Monitor closely and apply preventive fungicide within the next few days.",
        "description": (
            "Common rust is caused by the fungus <em>Puccinia sorghi</em>. "
            "It appears as small, circular to elongated golden-brown to brick-red pustules "
            "scattered on both leaf surfaces. Severe infections reduce photosynthesis and "
            "can lower yield by up to 20%."
        ),
        "symptoms": [
            "🔴 Small golden-brown to dark-red pustules on both leaf surfaces",
            "🟡 Yellow halo surrounding pustule clusters",
            "🍂 Heavy infection causes leaves to yellow and die early",
            "📉 Reduced photosynthesis and stunted plant growth",
        ],
        "treatment": [
            "💊 Apply triazole or strobilurin fungicides at first sign of infection",
            "🚫 Avoid excess nitrogen fertilisation which encourages lush growth",
            "🗑️ Remove and destroy heavily infected crop debris after harvest",
            "💧 Avoid overhead irrigation to reduce leaf wetness",
        ],
        "prevention": [
            "🌱 Use certified rust-resistant hybrid seed varieties",
            "🔄 Rotate crops each season to break the disease cycle",
            "👁️ Monitor fields regularly, especially during warm and humid weather",
            "📋 Keep field records to detect early spread patterns",
        ],
    },
    "Potato-Early_blight": {
        "plant":   "Potato",
        "disease": "Early Blight",
        "emoji":   "🥔",
        "severity": "High",
        "sev_class": "pd-danger",
        "action_urgency": "Treat <strong>immediately</strong> — this is a high-severity disease that spreads rapidly and can defoliate the entire plant.",
        "description": (
            "Early blight is caused by the fungus <em>Alternaria solani</em>. "
            "It is one of the most common potato diseases worldwide, producing characteristic "
            "dark brown lesions with concentric rings (target-board pattern). "
            "It thrives in warm, wet conditions and can defoliate plants if uncontrolled."
        ),
        "symptoms": [
            "🎯 Dark brown spots with concentric ring 'target-board' pattern",
            "🟡 Yellow chlorotic halo surrounding each lesion",
            "⬇️ Begins on older lower leaves and progresses upward",
            "🍂 Severe defoliation in advanced stages",
        ],
        "treatment": [
            "💊 Apply chlorothalonil, mancozeb, or copper-based fungicides",
            "⏱️ Start sprays before or at the very first symptom",
            "✂️ Remove and dispose of infected leaves immediately",
            "💧 Avoid overhead irrigation; use drip irrigation if possible",
        ],
        "prevention": [
            "🌱 Use certified disease-free seed tubers",
            "🧪 Maintain adequate potassium nutrition",
            "📏 Ensure good air circulation with proper plant spacing",
            "🔄 Rotate potatoes with non-host crops for at least 2 years",
        ],
    },
    "Tomato-Bacterial_spot": {
        "plant":   "Tomato",
        "disease": "Bacterial Spot",
        "emoji":   "🍅",
        "severity": "High",
        "sev_class": "pd-danger",
        "action_urgency": "Treat <strong>immediately</strong> — bacterial diseases spread through water and tools and can devastate the entire crop.",
        "description": (
            "Bacterial spot is caused by <em>Xanthomonas</em> species. "
            "It affects leaves, stems, and fruit, causing water-soaked lesions that turn "
            "dark brown to black with yellow halos. The disease spreads rapidly in warm, "
            "rainy weather and through contaminated tools and irrigation water."
        ),
        "symptoms": [
            "💧 Small water-soaked, greasy-looking spots on leaves",
            "⬛ Spots enlarge and turn dark brown/black with yellow borders",
            "🍅 Raised scab-like spots on fruit surface",
            "🍂 Defoliation in severe cases, exposing fruit to sun scald",
        ],
        "treatment": [
            "💊 Apply copper-based bactericides combined with mancozeb",
            "✂️ Remove and destroy infected plant material promptly",
            "🚶 Avoid working with plants when foliage is wet",
            "🧴 Disinfect tools with 10% bleach solution between uses",
        ],
        "prevention": [
            "🌱 Use certified disease-free or treated seed",
            "💧 Avoid overhead watering; water at base of plants only",
            "🔄 Rotate tomatoes with non-solanaceous crops",
            "🛡️ Plant resistant varieties where available",
        ],
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

# ── Session state ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Helper: render bullet list ──────────────────────────────────────────────────
def render_bullets(items: list[str]) -> str:
    rows = "".join(
        f'<div class="pd-bullet">{item}</div>' for item in items
    )
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Plant Disease Detector")
    st.markdown("---")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Min. confidence to accept result (%)",
        min_value=30, max_value=95, value=50, step=5,
        help="If the top prediction is below this, a low-confidence warning is shown.",
    )
    show_image_info = st.checkbox("Show image metadata", value=False)

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. Upload a **JPG, JPEG or PNG** leaf image
2. Click **🔬 Analyse**
3. View disease info & treatment advice
4. Check the **History** tab for past results
""")
    st.markdown("---")
    st.caption("Model: CNN · 3 classes · Input 256×256")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🌿 Plant Disease Detection System")
st.markdown(
    "Upload a clear photo of a **corn, potato, or tomato** leaf to identify diseases "
    "and get detailed treatment & prevention advice."
)
st.markdown("---")

tab_predict, tab_history, tab_about = st.tabs(
    ["🔍 Detect Disease", "📜 Prediction History", "ℹ️ About"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_up, col_res = st.columns([1, 1], gap="large")

    # ── Upload column ─────────────────────────────────────────────────────────
    with col_up:
        st.markdown("### 📷 Upload Leaf Image")
        plant_image = st.file_uploader(
            "leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="pd-upload-hint">Supported: JPG · JPEG · PNG</p>',
            unsafe_allow_html=True,
        )

        analyse_btn = False
        if plant_image:
            pil_img = Image.open(plant_image)
            # Convert to RGB to handle RGBA / palette PNGs safely
            pil_img_rgb = pil_img.convert("RGB")
            st.image(pil_img_rgb, caption="Uploaded Image", use_container_width=True)

            if show_image_info:
                st.markdown(
                    f'<div class="pd-card">'
                    f'<div class="pd-section-heading">🖼 Image Details</div>'
                    f'<div class="pd-label">Format: <strong>{pil_img.format or plant_image.type}</strong></div>'
                    f'<div class="pd-label">Dimensions: <strong>{pil_img.size[0]} × {pil_img.size[1]} px</strong></div>'
                    f'<div class="pd-label">Mode: <strong>{pil_img.mode}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            analyse_btn = st.button("🔬 Analyse", type="primary", use_container_width=True)
        else:
            st.markdown(
                '<div class="pd-empty">'
                '📂 <strong>No image uploaded</strong>'
                '<div class="pd-empty-sub">Upload a leaf photo above to begin detection.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Results column ────────────────────────────────────────────────────────
    with col_res:
        st.markdown("### 📊 Results")

        if plant_image and analyse_btn:
            with st.spinner("🔍 Analysing…"):
                # Decode with OpenCV
                plant_image.seek(0)
                raw = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                if bgr is None:
                    st.markdown(
                        '<div class="pd-danger">'
                        '❌ <strong>Could not read the image.</strong><br>'
                        'Please upload a valid JPG or PNG file and try again.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Pre-process
                    resized   = cv2.resize(bgr, (256, 256))
                    inp       = resized.reshape(1, 256, 256, 3).astype(np.float32) / 255.0
                    preds     = model.predict(inp, verbose=0)[0]
                    top_idx   = int(np.argmax(preds))
                    top_conf  = float(preds[top_idx]) * 100
                    result_key = CLASS_NAMES[top_idx]
                    info       = DISEASE_INFO[result_key]

                    # Save to history
                    st.session_state.history.append({
                        "time":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "filename":   plant_image.name,
                        "plant":      info["plant"],
                        "disease":    info["disease"],
                        "confidence": f"{top_conf:.1f}%",
                        "severity":   info["severity"],
                    })

                    # ── Low confidence warning ────────────────────────────────
                    if top_conf < confidence_threshold:
                        st.markdown(
                            f'<div class="pd-low-conf">'
                            f'⚠️ <strong>Low confidence ({top_conf:.1f}%)</strong><br>'
                            f'The model is not confident enough in this result. '
                            f'Please try a clearer, well-lit photo focused on the leaf surface.'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Main result card ──────────────────────────────────────
                    sev_icon = "🟡" if info["severity"] == "Moderate" else "🔴"
                    st.markdown(
                        f'<div class="pd-card">'
                        f'  <div class="pd-plant-name">{info["emoji"]} {info["plant"]}</div>'
                        f'  <div class="pd-disease-name">🦠 {info["disease"]}</div>'
                        f'  <div class="pd-meta">'
                        f'    {sev_icon} Severity: <strong>{info["severity"]}</strong>'
                        f'    &nbsp;|&nbsp; 🎯 Confidence: <strong>{top_conf:.1f}%</strong>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if top_conf >= confidence_threshold:
                        st.markdown("---")

                        # Description
                        st.markdown(
                            '<div class="pd-section-heading">📝 About this disease</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="pd-info">{info["description"]}</div>',
                            unsafe_allow_html=True,
                        )

                        # Symptoms
                        with st.expander("🔎 Symptoms", expanded=True):
                            st.markdown(render_bullets(info["symptoms"]), unsafe_allow_html=True)

                        # Treatment
                        with st.expander("💊 Treatment", expanded=True):
                            st.markdown(render_bullets(info["treatment"]), unsafe_allow_html=True)

                        # Prevention
                        with st.expander("🛡️ Prevention", expanded=False):
                            st.markdown(render_bullets(info["prevention"]), unsafe_allow_html=True)

                        # Action urgency box
                        st.markdown(
                            f'<div class="{info["sev_class"]}">'
                            f'⚡ <strong>Recommended action:</strong><br>'
                            f'{info["action_urgency"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        elif not plant_image:
            st.markdown(
                '<div class="pd-empty">'
                '🔬 <strong>Results will appear here</strong>'
                '<div class="pd-empty-sub">Upload an image and click Analyse.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📜 Prediction History")

    if not st.session_state.history:
        st.markdown(
            '<div class="pd-empty">'
            '📂 <strong>No predictions yet</strong>'
            '<div class="pd-empty-sub">Go to the Detect Disease tab to analyse a leaf image.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.rerun()

        # Stats row
        total  = len(st.session_state.history)
        high_s = sum(1 for h in st.session_state.history if h["severity"] == "High")
        diseases_list = [h["disease"] for h in st.session_state.history]
        most_common = max(set(diseases_list), key=diseases_list.count)

        c1, c2, c3 = st.columns(3)
        for col, num, label in [
            (c1, str(total),       "Total Analyses"),
            (c2, most_common,      "Most Detected"),
            (c3, str(high_s),      "High Severity Cases"),
        ]:
            with col:
                st.markdown(
                    f'<div class="pd-stat-box">'
                    f'  <div class="pd-stat-num">{num}</div>'
                    f'  <div class="pd-stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # History rows (newest first)
        for entry in reversed(st.session_state.history):
            sev_color = "#c62828" if entry["severity"] == "High" else "#f9a825"
            st.markdown(
                f'<div class="pd-hist-row">'
                f'  🕐 <strong>{entry["time"]}</strong> &nbsp;|&nbsp; '
                f'  📁 {entry["filename"]} &nbsp;|&nbsp; '
                f'  🌿 {entry["plant"]} &nbsp;|&nbsp; '
                f'  🦠 {entry["disease"]} &nbsp;|&nbsp; '
                f'  🎯 {entry["confidence"]} &nbsp;|&nbsp; '
                f'  <span style="color:{sev_color};font-weight:700;">⚠ {entry["severity"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About this Application")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            '<div class="pd-about-box">'
            '<h4>🧠 Model Details</h4>'
            '<ul>'
            '<li>Architecture: Convolutional Neural Network (CNN)</li>'
            '<li>Input size: 256 × 256 × 3 (RGB)</li>'
            '<li>Output: 3 classes (softmax)</li>'
            '<li>Framework: Keras / TensorFlow 2.x</li>'
            '</ul>'
            '<h4>🌿 Supported Diseases</h4>'
            '<table style="width:100%;border-collapse:collapse;">'
            '<tr style="background:#e8f5e9;">'
            '  <th style="padding:6px 10px;text-align:left;color:#1b5e20;">Plant</th>'
            '  <th style="padding:6px 10px;text-align:left;color:#1b5e20;">Disease</th>'
            '  <th style="padding:6px 10px;text-align:left;color:#1b5e20;">Severity</th>'
            '</tr>'
            '<tr><td style="padding:6px 10px;">🌽 Corn</td><td style="padding:6px 10px;">Common Rust</td><td style="padding:6px 10px;color:#f9a825;">Moderate</td></tr>'
            '<tr style="background:#f9f9f9;"><td style="padding:6px 10px;">🥔 Potato</td><td style="padding:6px 10px;">Early Blight</td><td style="padding:6px 10px;color:#c62828;">High</td></tr>'
            '<tr><td style="padding:6px 10px;">🍅 Tomato</td><td style="padding:6px 10px;">Bacterial Spot</td><td style="padding:6px 10px;color:#c62828;">High</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            '<div class="pd-about-box">'
            '<h4>📌 Tips for Best Results</h4>'
            '<ul>'
            '<li>Use a <strong>clear, well-lit</strong> photo of a single leaf</li>'
            '<li>Fill the frame with the leaf — avoid too much background</li>'
            '<li>Use natural daylight for best colour accuracy</li>'
            '<li>Avoid blurry or heavily shadowed images</li>'
            '<li>Upload <strong>JPG, JPEG or PNG</strong> files only</li>'
            '</ul>'
            '<h4>⚠️ Disclaimer</h4>'
            '<p style="color:#555555;">'
            'This tool is intended as a <strong>decision-support aid</strong> only. '
            'It detects only the 3 diseases it was trained on. '
            'Always consult a qualified agronomist or plant pathologist '
            'for a confirmed diagnosis and treatment plan.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("🌿 Plant Disease Detection System · Keras / TensorFlow · Streamlit")
