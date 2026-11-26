import tensorflow as tf
import json
from pathlib import Path

# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

DATASET_PATH = Path("dataset") / "plantvillage dataset" / "color"
MODEL_DIR = Path("models")

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
NUM_CLASSES = 38

MODEL_DIR.mkdir(exist_ok=True)

# =====================================
# 2. LOAD TRAINING DATA
# =====================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# =====================================
# 3. LOAD VALIDATION DATA
# =====================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names

print("\nTotal classes:", len(class_names))

# Save class names for future predictions
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# =====================================
# 4. LOAD PRETRAINED MOBILENETV2
# =====================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================
# 5. BUILD CLASSIFICATION MODEL
# =====================================

inputs = tf.keras.Input(shape=(160, 160, 3))

# Prepare image pixels
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# Extract image features
x = base_model(x, training=False)

# Reduce feature dimensions
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Reduce overfitting
x = tf.keras.layers.Dropout(0.2)(x)

# Classify into 38 categories
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

# Create final model
model = tf.keras.Model(inputs, outputs)

# =====================================
# 6. COMPILE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# 7. TRAINING CALLBACKS
# =====================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_DIR / "plantguard_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =====================================
# 8. TRAIN MODEL
# =====================================

print("\nStarting PlantGuard AI Training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    callbacks=[early_stopping, model_checkpoint]
)

# =====================================
# 9. SAVE FINAL MODEL
# =====================================

model.save(MODEL_DIR / "plantguard_model.keras")

print("\nTraining Completed Successfully!")
print("Model saved inside models folder.")