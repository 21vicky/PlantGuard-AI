import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# =====================================
# 1. CONFIGURATION
# =====================================

MODEL_PATH = Path("models") / "plantguard_best.keras"
CLASS_PATH = Path("models") / "class_names.json"

IMAGE_SIZE = (160, 160)

# =====================================
# 2. LOAD MODEL AND CLASSES
# =====================================

print("Loading PlantGuard AI...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("Model loaded successfully!")

# =====================================
# 3. IMAGE PREPROCESSING
# =====================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# =====================================
# 4. PREDICTION FUNCTION
# =====================================

def predict_plant_disease(image_path):

    image_array = preprocess_image(image_path)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    print("\n🌱 PLANTGUARD AI RESULT")
    print("-" * 40)

    print("Predicted Class:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

    return predicted_class, confidence

# =====================================
# 5. TEST PREDICTION
# =====================================

if __name__ == "__main__":

    image_path = input("\nEnter leaf image path: ").strip().strip('"')

    predict_plant_disease(image_path)