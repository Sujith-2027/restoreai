# ReStoreAI 🔧

> AI-Powered Device Damage Analysis & Intelligent Repair Recommendation Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-97.03%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is ReStoreAI?

ReStoreAI is an AI-powered web application that automatically classifies electronic devices from images, analyzes structural damage severity, estimates repair costs, and recommends the nearest repair shops on an interactive map — all in under 2 seconds.

---

## Features

- **AI Device Classification** — MobileNetV2 deep learning model trained to 97.03% accuracy across 6 device categories
- **Damage Analysis** — Detects cracks, rust, and broken components from uploaded images
- **Repair Cost Estimation** — Provides min/max cost range based on damage severity
- **Live Repair Shop Finder** — Uses TomTom API to find real nearby repair shops based on GPS location
- **Interactive Map** — Dark-themed map with numbered markers for the 3 nearest shops
- **PDF Report** — Downloadable analysis report with full device summary
- **Analytics Dashboard** — Visual insights on analysis history and repairability trends

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI Model | TensorFlow, Keras, MobileNetV2 |
| Frontend | HTML, CSS, JavaScript |
| Maps | Leaflet.js + TomTom Search API |
| Image Processing | Pillow, NumPy |
| Deployment | Render |

---

## Device Categories

The model classifies 6 electronic device types:
- Smartphone / Mobile Phone
- Laptop
- Television
- Refrigerator
- Air Conditioner
- Washing Machine

---

## Project Structure

```
ReStoreAI/
├── app_final.py          # Main Flask application
├── setup_map.py          # Downloads Leaflet map files
├── requirements.txt      # Python dependencies
├── render.yaml           # Render deployment config
│
├── templates/            # HTML templates
│   ├── base.html
│   ├── home.html
│   ├── analyze.html
│   └── result.html
│
├── static/               # Leaflet JS/CSS files
├── models/               # Trained MobileNetV2 weights (.h5)
├── uploads/              # Temporary image uploads
└── reports/              # Generated PDF reports
```

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Sujith-2027/restoreai.git
cd restoreai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your TomTom API key**

In `app_final.py` line 39:
```python
TOMTOM_API_KEY = "your_key_here"
```
Get a free key (no credit card) at [developer.tomtom.com](https://developer.tomtom.com)

**4. Download map files**
```bash
python setup_map.py
```

**5. Run the app**
```bash
python app_final.py
```

Visit `http://localhost:5000`

---

## How It Works

```
User uploads image
        ↓
MobileNetV2 classifies device (97% accuracy)
        ↓
Damage analysis (cracks / rust / broken %)
        ↓
Repair cost estimation
        ↓
GPS location → TomTom API → 3 nearest repair shops
        ↓
Interactive map + downloadable PDF report
```

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 97.03% |
| Architecture | MobileNetV2 |
| Input Size | 224 x 224 px |
| Classes | 6 |
| Training Platform | TensorFlow / Keras |

---

## Author

**Sujith** — [GitHub](https://github.com/Sujith-2027)

---

*Built for academic submission — ReStoreAI © 2026*