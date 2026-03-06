"""
ReStoreAI Web Application - MobileNetV2 Version
Simple Flask deployment for device classification

Usage:
    pip install flask
    python app_mobilenet.py
    
Then open: http://localhost:5000
"""

from flask import Flask, request, render_template_string, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs('uploads', exist_ok=True)

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 6

# Device information with repair costs
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

# HTML Template with beautiful design
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ReStoreAI - AI Device Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .accuracy-badge {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            text-align: center;
            margin-bottom: 30px;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            padding: 60px;
            text-align: center;
            border-radius: 15px;
            margin: 30px 0;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .upload-area:hover {
            background: #e8edff;
            border-color: #5568d3;
        }
        
        .upload-area h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .upload-area p {
            color: #888;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 40px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 18px;
            width: 100%;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        #preview {
            max-width: 100%;
            max-height: 400px;
            margin: 20px auto;
            display: block;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .result {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .device-name {
            font-size: 32px;
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 24px;
            color: #38ef7d;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .cost {
            font-size: 28px;
            color: #2ecc71;
            font-weight: bold;
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 10px;
        }
        
        .info-section {
            margin-top: 20px;
            padding: 20px;
            background: white;
            border-radius: 10px;
        }
        
        .info-section h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .issues li {
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .issues li:before {
            content: "⚠️";
            position: absolute;
            left: 0;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #888;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 ReStoreAI</h1>
        <p class="subtitle">AI-Powered Device Classification & Repair Cost Estimation</p>
        <div class="accuracy-badge">
            🎯 Model Accuracy: 97%+ | Powered by MobileNetV2
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📸 Upload Device Image</h3>
                <p>Click to select or drag and drop</p>
                <p style="font-size: 0.9em; margin-top: 10px;">Supports: TV, Laptop, Fridge, AC, Washing Machine, Mobile/Tablet</p>
                <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;" onchange="previewImage(this)">
            </div>
            
            <img id="preview" style="display: none;">
            
            <button type="submit">🚀 Analyze Device</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image with AI...</p>
        </div>
        
        <div id="result"></div>
        
        <div class="footer">
            Powered by TensorFlow & MobileNetV2 | ReStoreAI © 2026
        </div>
    </div>
    
    <script>
        function previewImage(input) {
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                }
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.error) {
                    result.innerHTML = `<div class="result" style="border-left-color: #e74c3c;"><p style="color: #e74c3c;">❌ ${data.error}</p></div>`;
                    return;
                }
                
                result.innerHTML = `
                    <div class="result">
                        <div class="device-name">🔍 ${data.device}</div>
                        <div class="confidence">✅ Confidence: ${data.confidence}%</div>
                        <div class="cost">💰 Estimated Repair Cost: ₹${data.cost_min.toLocaleString()} - ₹${data.cost_max.toLocaleString()}</div>
                        
                        <div class="info-section">
                            <h4>📊 Device Information</h4>
                            <p><strong>⏳ Average Lifespan:</strong> ${data.lifespan}</p>
                        </div>
                        
                        <div class="info-section">
                            <h4>⚠️ Common Issues</h4>
                            <ul class="issues">
                                ${data.issues.map(issue => `<li>${issue}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
            } catch (error) {
                loading.style.display = 'none';
                result.innerHTML = `<div class="result" style="border-left-color: #e74c3c;"><p style="color: #e74c3c;">❌ Error: ${error.message}</p></div>`;
            }
        };
    </script>
</body>
</html>
'''

# Global model variable
model = None

def create_model_architecture():
    """Create MobileNetV2 model architecture"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
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
    
    return Model(inputs, outputs, name='MobileNetV2_DeviceClassifier')

def load_model():
    """Load trained MobileNetV2 model"""
    global model
    
    if model is not None:
        return model
    
    print("Loading MobileNetV2 model...")
    
    # Find best weights
    weights_path = 'models/best_mobilenet_weights.h5'
    
    if not os.path.exists(weights_path):
        raise Exception(f"Model weights not found at {weights_path}!")
    
    # Create model and load weights
    model = create_model_architecture()
    
    # Build model with dummy input
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype('float32')
    _ = model(dummy_input)
    
    # Load weights
    model.load_weights(weights_path)
    
    print(f"✅ Model loaded successfully from {weights_path}")
    return model

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load model
        model = load_model()
        
        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        class_names = list(DEVICE_INFO.keys())
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx]) * 100
        device_key = class_names[pred_idx]
        
        # Get device info
        info = DEVICE_INFO[device_key]
        min_cost, max_cost = info['repair_cost_range']
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'device': info['display_name'],
            'confidence': f'{confidence:.1f}',
            'cost_min': min_cost,
            'cost_max': max_cost,
            'lifespan': info['avg_lifespan'],
            'issues': info['common_issues']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("ReStoreAI Web Application - MobileNetV2")
    print("="*80)
    print("\nStarting server...")
    print("✅ Model: MobileNetV2 (97%+ accuracy)")
    print("✅ Classes: 6 device types")
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)