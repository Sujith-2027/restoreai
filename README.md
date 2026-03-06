# ReStore AI - Setup (Local)

1. Create venv:
   python3 -m venv venv
   source venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. Copy .env.example -> .env and fill values:
   - DATABASE_URL
   - SERPAPI_KEY (optional)
   - MODEL_PATH (optional)
   - GST_RATE, LABOUR_BASE etc.

4. Create uploads folder:
   mkdir -p uploads

5. Initialize DB tables (if using Postgres):
   - Run the SQL in README or let the app attempt to insert (create table with provided SQL recommended).

6. Start the app:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

7. Open:
   http://127.0.0.1:8000/upload

# ReStoreAI - AI Device Classification System

## 🎯 Project Overview
AI-powered device classification system with 97.03% accuracy for automatic device identification, repair cost estimation, and repair/recycle recommendations with nearby location suggestions.

## ⭐ Key Features
- **97.03% Classification Accuracy** using MobileNetV2
- **6 Device Categories**: TV, Laptop, Fridge, AC, Washing Machine, Mobile/Tablet
- **Smart Recommendations**: Repair vs Recycle based on probability
- **Cost Estimation**: Repair cost ranges for each device
- **Location Integration**: Nearby repair shops or recycling centers with Google Maps
- **Beautiful UI**: Responsive purple gradient design

## 📊 Technical Specifications
- **Model**: MobileNetV2 (2.6M parameters)
- **Training Accuracy**: 96.97%
- **Validation Accuracy**: 97.03%
- **Dataset**: 1,525 images (1,222 train + 303 validation)
- **Framework**: TensorFlow 2.10 + Keras
- **Web Framework**: Flask
- **Inference Time**: <1 second

## 🚀 Quick Start

### Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure model file exists
# models/best_mobilenet_weights.h5 should be present

# 3. Run the application
python app_final.py
```

### Access
Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure
```
ReStoreAI/
├── templates/
│   ├── base.html          # Base template with navigation
│   ├── home.html          # Landing page
│   ├── analyze.html       # Upload page
│   ├── result.html        # Results with recommendations
│   └── report.html        # Model performance report
├── static/
│   └── uploads/           # Uploaded images
├── models/
│   └── best_mobilenet_weights.h5  # Trained model (97% accuracy)
├── app_final.py           # Main Flask application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 User Interface Pages

### 1. Home Page
- Hero section with features
- Statistics (97% accuracy, 6 categories)
- Feature cards
- Call-to-action

### 2. Analyze Page
- Drag & drop image upload
- Supported devices display
- File preview
- Beautiful upload interface

### 3. Result Page
- Device classification with confidence
- Repair vs Recycle recommendation
- Cost estimation
- Common issues list
- Nearby repair shops or recycling centers
- Google Maps integration

### 4. Report Page
- Model performance metrics
- Architecture diagram
- Technical specifications
- Supported devices

## 🔧 How It Works

1. **Upload**: User uploads device image
2. **Classification**: MobileNetV2 predicts device type (97% accuracy)
3. **Analysis**: 
   - Calculates repair probability from confidence score
   - Determines repair vs recycle recommendation
4. **Results**: Shows:
   - Device type & confidence
   - Repair probability
   - Cost estimation
   - Nearby locations (repair shops or recycling centers)
   - Common issues

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.03% |
| Training Accuracy | 96.97% |
| Parameters | 2.6M |
| Inference Time | <1s |
| Input Size | 224×224×3 |

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow 2.10
- **Model**: MobileNetV2 (ImageNet pretrained)
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: Pillow (PIL)
- **Deployment**: Python 3.10

## 📝 API Endpoints

- `GET /` - Home page
- `GET /analyze` - Upload page
- `POST /analyze` - Process uploaded image
- `GET /report` - Model performance report

## 🎯 Repair Probability Logic

```python
repair_probability = confidence * 0.85 + 10

if repair_probability > 60:
    recommendation = "repair"
    show = "nearby repair shops"
else:
    recommendation = "recycle"
    show = "nearby recycling centers"
```

## 🗺️ Location Features

- Shows 3 nearby locations based on recommendation
- Google Maps integration for directions
- Distance, ratings, and reviews
- "View All on Maps" button

## 📱 Supported Devices

1. **Television** - All types (LED, LCD, Smart TV)
2. **Laptop** - Notebooks, Ultrabooks
3. **Refrigerator** - All models
4. **Air Conditioner** - Split, Window, Portable
5. **Washing Machine** - Front & Top load
6. **Mobile/Tablet** - Smartphones, Tablets

## 🎓 Academic Use

Submitted by: [Your Name]
Project: ReStoreAI
Date: February 15, 2026
Accuracy: 97.03%

## 📄 License

Academic Project - All Rights Reserved

## 🙏 Acknowledgments

- TensorFlow & Keras teams
- MobileNetV2 architecture
- ImageNet dataset
- Flask framework

---

**Built with 💜 using AI & Python**