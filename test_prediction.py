# test_prediction.py
import json, numpy as np, tensorflow as tf
from PIL import Image

MODEL_PATH = "models/image_model.h5"
CLASS_MAP_PATH = "models/class_indices.json"
TEST_IMAGE = r"C:\Users\91989\Downloads\ReStoreAI\dataset\Fridge\fridge (24).png"
  # change to your test image path

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)
idx2label = {v:k for k,v in class_map.items()}

img = Image.open(TEST_IMAGE).convert("RGB").resize((224,224))
x = np.expand_dims(np.array(img)/255.0, axis=0)
pred = model.predict(x)
idx = int(pred.argmax())
print("Predicted:", idx2label.get(idx, "Unknown"), "conf:", float(pred.max()))
