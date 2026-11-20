<div align="center">

# 🌿 PlantVision — Disease Detector

### AI-powered plant disease detection from leaf images

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plantvision-disease-detector-lekhraj04.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.15-D00000?style=flat&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

<br/>

> Upload a leaf photo → get an instant diagnosis, treatment steps, and prevention advice.

<br/>

---

</div>

## 📌 Overview

**PlantVision** is a deep learning web application that detects plant diseases from leaf images in real time. It uses a Convolutional Neural Network (CNN) trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and serves predictions through an interactive Streamlit interface.

The app gives farmers, agronomists, and researchers an instant, actionable diagnosis — including the disease description, symptom checklist, treatment plan, and urgency level — all from a single photo.

---

## 🎯 Features

| Feature | Details |
|---|---|
| 🔍 **Instant Detection** | Upload a leaf image and get a diagnosis in under 2 seconds |
| 📋 **Disease Details** | Full description with pathogen name, severity level, and confidence score |
| 💊 **Treatment Guide** | Step-by-step treatment recommendations for each disease |
| 🛡️ **Prevention Tips** | Season-ahead prevention strategies to protect future crops |
| ⚡ **Urgency Alerts** | Colour-coded action box (Moderate / High) for quick triage |
| 📜 **Prediction History** | Session history with stats — total analyses, most detected disease |
| 🌑 **Dark & Light Mode** | Full theme compatibility — text always readable in both modes |
| 📱 **Responsive Layout** | Wide layout that works on desktop and tablet screens |

---

## 🌿 Supported Plants & Diseases

| # | Plant | Disease | Severity | Pathogen |
|---|-------|---------|----------|----------|
| 1 | 🌽 Corn (Maize) | Common Rust | 🟡 Moderate | *Puccinia sorghi* |
| 2 | 🥔 Potato | Early Blight | 🔴 High | *Alternaria solani* |
| 3 | 🍅 Tomato | Bacterial Spot | 🔴 High | *Xanthomonas* spp. |

---

## 🧠 Model Architecture

```
Input (256 × 256 × 3)
    │
    ├── Conv2D (32 filters, 3×3, ReLU)
    ├── MaxPooling2D (3×3)
    ├── Conv2D (16 filters, 3×3, ReLU)
    ├── MaxPooling2D (2×2)
    ├── Flatten
    ├── Dense (8 units, ReLU)
    └── Dense (3 units, Softmax)  →  [Corn Rust | Potato Blight | Tomato Spot]
```

- **Framework:** Keras / TensorFlow 2.21
- **Input size:** 256 × 256 × 3 (RGB)
- **Output:** 3-class softmax probability
- **Model file:** `plant_disease.h5` (2.7 MB)

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Lekhrajpatil/PlantVision-Disease-Detector.git
cd PlantVision-Disease-Detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run main_app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
PlantVision-Disease-Detector/
│
├── main_app.py              # Streamlit application — UI + prediction logic
├── plant_disease.h5         # Trained CNN model weights (Keras HDF5)
├── requirements.txt         # Python dependencies with pinned versions
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🖥️ How to Use

1. Open the app at the live URL or run it locally
2. Go to the **🔍 Detect Disease** tab
3. Upload a **JPG, JPEG, or PNG** photo of a plant leaf
4. Click **🔬 Analyse**
5. Read the diagnosis, confidence score, and treatment advice
6. Check the **📜 Prediction History** tab to review past analyses

> **Tip:** Use a clear, well-lit, close-up photo of a single leaf for the most accurate result.

---

## 📸 Screenshots

| Upload & Detect | Disease Details |
|---|---|
| Upload a leaf image and click Analyse | Full diagnosis with treatment & prevention |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Framework** | TensorFlow 2.21 · Keras 3.15 |
| **Image Processing** | OpenCV 5.1 · Pillow 13.0 |
| **Web Framework** | Streamlit 1.67 |
| **Language** | Python 3.10+ |
| **Deployment** | Streamlit Community Cloud |

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud** (free tier).

To deploy your own fork:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your fork
4. Set **Main file path** to `main_app.py`
5. Click **Deploy**

---

## ⚠️ Disclaimer

PlantVision is a **decision-support tool** only. It currently detects the 3 diseases it was trained on. Always consult a qualified agronomist or plant pathologist for a confirmed field diagnosis and treatment plan.

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

Made with 🌿 by [Lekhraj Patil](https://github.com/Lekhrajpatil)

</div>
