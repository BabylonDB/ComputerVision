import gradio as gr
from transformers import pipeline
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib

# Laden der Klassen-Liste aus Training
class_names = joblib.load("class_names.pkl")  # <- hier deine Label-Datei

# Dein Modell (muss exakt wie beim Training aussehen!)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')  # jetzt flexibel!
])

# Lade die Gewichte aus .h5
model.load_weights("watch_brand_classifier.h5")

# CLIP Zero-Shot
clip_detector = pipeline(
    model="openai/clip-vit-large-patch14",
    task="zero-shot-image-classification"
)

# Inferenzfunktion
def classify_watch(image):
    if image is None:
        return "No image provided", {}

    # Resize/Normalize
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # CNN Prediction
    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    # Fehlerabfang falls irgendwas schief läuft
    if top_index >= len(class_names):
        return {
            "Custom Trained Model Prediction": {
                "label": "Unknown (IndexError)",
                "confidence": float(preds[top_index])
            },
            "CLIP Zero-Shot Prediction": {}
        }

    custom_model_prediction = {
        "label": class_names[top_index],
        "confidence": float(preds[top_index])
    }

    # CLIP Prediction
    clip_results = clip_detector(image, candidate_labels=list(class_names))
    clip_output = {res["label"]: float(res["score"]) for res in clip_results}

    return {
        "Custom Trained Model Prediction": custom_model_prediction,
        "CLIP Zero-Shot Prediction": clip_output
    }

# Gradio Interface
iface = gr.Interface(
    fn=classify_watch,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Watch Brand Classifier",
    description="Upload a watch image to compare predictions from a trained CNN and a zero-shot CLIP model.",
    examples=[
        ["https://huggingface.co/spaces/BerkantBaskaya/Computer_Vision_Watch/resolve/main/0.jpg"],
        ["https://huggingface.co/spaces/BerkantBaskaya/Computer_Vision_Watch/resolve/main/1.jpg"],
        ["https://huggingface.co/spaces/BerkantBaskaya/Computer_Vision_Watch/resolve/main/2.jpg"]
    ]
)

iface.launch()
