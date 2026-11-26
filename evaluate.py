import tensorflow as tf
from pathlib import Path

# Project configuration
DATASET_PATH = Path("dataset") / "plantvillage dataset" / "color"
MODEL_PATH = Path("models") / "plantguard_best.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nPlantGuard AI Model Evaluation")
print("-" * 40)

# Evaluate model
loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\nEvaluation Results")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")