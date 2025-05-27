import gradio as gr
from transformers import pipeline
from PIL import Image

# Load models
vit_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
clip_detector = pipeline(model="openai/clip-vit-large-patch14", task="zero-shot-image-classification")

# Define watch brand labels
labels_watch_brands = [
    'Casio', 'Fossil', 'Guess', 'Versace', 'Armani Exchange',
    'Michael Kors', 'Tissot', 'Seiko', 'Citizen', 'Tommy Hilfiger',
    'TW Steel', 'BOSS', 'Bellroy', 'iConnect By Timex', 'August Berg'
]

# Inference function
def classify_watch(image):
    vit_results = vit_classifier(image)
    vit_output = {result['label']: result['score'] for result in vit_results}

    clip_results = clip_detector(image, candidate_labels=labels_watch_brands)
    clip_output = {result['label']: result['score'] for result in clip_results}

    return {
        "ViT Classification": vit_output,
        "CLIP Zero-Shot Classification": clip_output
    }

# Example images
example_images = [
    ["images/0.jpg"],
    ["images/1.jpg"],
    ["images/2.jpg"]
]

# Gradio interface
iface = gr.Interface(
    fn=classify_watch,
    inputs=gr.Image(type="filepath"),
    outputs=gr.JSON(),
    title="Watch Brand Classifier",
    description="Upload a watch image to compare predictions from a pretrained ViT model and zero-shot CLIP.",
    examples=example_images
)

iface.launch()
