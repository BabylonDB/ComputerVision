import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Rebuild your CNN architecture exactly as used during training
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='softmax')  # Make sure this matches your class count
])

# Load only the weights from the .h5 file
model.load_weights("watch_brand_classifier.h5")

# Class names (must match training order)
class_names = [
    'Armani Exchange', 'August Berg', 'BOSS', 'Bellroy', 'Casio',
    'Citizen', 'Fossil', 'Guess', 'Michael Kors', 'Seiko',
    'Tissot', 'Tommy Hilfiger', 'TW Steel', 'Versace', 'iConnect By Timex'
]

# Load CLIP model for zero-shot
clip_detector = pipeline(
    model="openai/clip-vit-large-patch14", 
    task="zero-shot-image-classification"
)

# Define inference function
def classify_watch(image):
    if image is None:
        return "No image provided", {}

    # Resize and normalize
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using custom model
    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    custom_model_prediction = {
        "label": class_names[top_index],
        "confidence": float(preds[top_index])
    }

    # Predict using CLIP
    clip_results = clip_detector(image, candidate_labels=class_names)
    clip_output = {res["label"]: float(res["score"]) for res in clip_results}

    return {
        "Custom Trained Model Prediction": custom_model_prediction,
        "CLIP Zero-Shot Prediction": clip_output
    }

# Gradio interface
iface = gr.Interface(
    fn=classify_watch,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Watch Brand Classifier",
    description="Upload a watch image to compare predictions from a trained CNN and a zero-shot CLIP model.",
    examples=[
        ["images/0.jpg"],
        ["images/1.jpg"],
        ["images/2.jpg"]
    ]
)

iface.launch()
