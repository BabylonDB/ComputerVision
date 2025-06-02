import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load your trained model
model = load_model("watch_brand_classifier.h5")  # Make sure this file is present

# Class names (must match your training labels)
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

    # Resize image and prepare for trained model
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with your trained model
    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    custom_model_prediction = {
        "label": class_names[top_index],
        "confidence": float(preds[top_index])
    }

    # Zero-shot CLIP prediction
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
