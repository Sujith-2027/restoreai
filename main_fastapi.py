"""
ReStoreAI - FastAPI Version (Compatible with your existing ml_utils.py)
97% Accuracy Device Classification
"""

import os
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 6

DEVICE_INFO = {
    "Air_Conditioner": {
        "display_name": "Air Conditioner",
        "repair_cost_range": (3000, 20000),
        "common_issues": ["Compressor failure", "Gas leakage", "Cooling issues"],
        "avg_lifespan": "10-15 years"
    },
    "Fridge": {
        "display_name": "Refrigerator",
        "repair_cost_range": (2500, 18000),
        "common_issues": ["Compressor failure", "Cooling issues", "Ice buildup"],
        "avg_lifespan": "10-15 years"
    },
    "Laptop": {
        "display_name": "Laptop",
        "repair_cost_range": (2000, 25000),
        "common_issues": ["Screen damage", "Battery failure", "Keyboard issues"],
        "avg_lifespan": "4-6 years"
    },
    "Mobile_Tablet": {
        "display_name": "Mobile/Tablet",
        "repair_cost_range": (1000, 10000),
        "common_issues": ["Screen damage", "Battery issues", "Charging port"],
        "avg_lifespan": "3-5 years"
    },
    "Television": {
        "display_name": "Television",
        "repair_cost_range": (2000, 15000),
        "common_issues": ["Screen damage", "Power supply failure", "Backlight issues"],
        "avg_lifespan": "7-10 years"
    },
    "Washing_machine": {
        "display_name": "Washing Machine",
        "repair_cost_range": (2000, 15000),
        "common_issues": ["Motor failure", "Drainage issues", "Drum problems"],
        "avg_lifespan": "8-12 years"
    }
}

model = None

def create_model_architecture():
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet', alpha=1.0)
    base_model.trainable = False
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs, outputs)

def load_model():
    global model
    if model is not None:
        return model
    print("Loading MobileNetV2 model...")
    model = create_model_architecture()
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype('float32')
    _ = model(dummy)
    model.load_weights('models/best_mobilenet_weights.h5')
    print("✅ Model loaded (97.03% accuracy)")
    return model

def preprocess_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "page": "home"})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "page": "analyze"})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, image: UploadFile = File(...)):
    try:
        # Save uploaded file
        upload_dir = Path("static/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / image.filename
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Load model and predict
        model = load_model()
        img_array = preprocess_image(str(file_path))
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        class_names = list(DEVICE_INFO.keys())
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx]) * 100
        device_key = class_names[pred_idx]
        info = DEVICE_INFO[device_key]
        
        # Repair probability
        repair_probability = round(confidence * 0.85 + 10)
        if repair_probability > 95:
            repair_probability = 95
        recommendation = 'repair' if repair_probability > 60 else 'recycle'
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "page": "result",
            "image_path": str(file_path),
            "result": info['display_name'],
            "confidence": round(confidence, 2),
            "cost_min": info['repair_cost_range'][0],
            "cost_max": info['repair_cost_range'][1],
            "lifespan": info['avg_lifespan'],
            "issues": info['common_issues'],
            "repair_probability": round(repair_probability, 1),
            "recommendation": recommendation
        })
    
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/report", response_class=HTMLResponse)
async def report(request: Request):
    return templates.TemplateResponse("report.html", {"request": request, "page": "report"})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "MobileNetV2", "accuracy": "97.03%"}

if __name__ == "__main__":
    import uvicorn
    print("="*80)
    print("ReStoreAI - FastAPI Server")
    print("="*80)
    print("\n✅ Model: MobileNetV2 (97.03% accuracy)")
    print("🌐 Server: http://localhost:8000")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)