import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
from dog_training import make_dog_model
from food_training import make_food_model

# ── load models ───────────────────────────────────────────────────────────────

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model.eval()

custom_model = make_dog_model()
custom_model.load_state_dict(torch.load('model_saver/best_model.pth', map_location='cpu'))
custom_model.eval()

food_model = make_food_model()
food_model.load_state_dict(torch.load('model_saver/best_food.pth', map_location='cpu'))
food_model.eval()

# ── preprocessing ─────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── labels ────────────────────────────────────────────────────────────────────

with open("imagenet_classes.txt") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]
with open("dog_labels.txt") as f:
    dog_labels = [s.strip() for s in f.readlines()]
with open("food101_labels.txt") as f:
    food_labels = [s.strip() for s in f.readlines()]

# ── inference ─────────────────────────────────────────────────────────────────

def classify(image, model, labels):
    img = preprocess(image.convert("RGB"))
    batch = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = model(batch)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_ids = torch.topk(probs, 5)
    return {labels[i.item()]: float(p) for i, p in zip(top_ids, top_probs)}

def classify_resnet(image):
    return classify(image, resnet_model, imagenet_labels)

def classify_dog(image):
    return classify(image, custom_model, dog_labels)

def classify_food(image):
    return classify(image, food_model, food_labels)

# ── interface ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-Model Image Classifier") as demo:
    gr.Markdown("""
# Multi-Model Image Classifier
Upload a photo or use your webcam. Switch between models using the tabs.
""")

    with gr.Tabs():
        with gr.Tab("ResNet-50 — General"):
            gr.Markdown("Pretrained on ImageNet — recognizes 1,000 everyday objects.")
            with gr.Row():
                img_r = gr.Image(type="pil", sources=["upload", "webcam"], label="Image")
                out_r = gr.Label(num_top_classes=5, label="Top 5 predictions")
            gr.Button("Classify").click(classify_resnet, inputs=img_r, outputs=out_r)

        with gr.Tab("Dog Breeds"):
            gr.Markdown("Fine-tuned ResNet-50 — identifies 37 dog breeds.")
            with gr.Row():
                img_d = gr.Image(type="pil", sources=["upload", "webcam"], label="Image")
                out_d = gr.Label(num_top_classes=5, label="Top 5 predictions")
            gr.Button("Classify").click(classify_dog, inputs=img_d, outputs=out_d)

        with gr.Tab("Food"):
            gr.Markdown("Fine-tuned ResNet-50 — classifies 101 food categories.")
            with gr.Row():
                img_f = gr.Image(type="pil", sources=["upload", "webcam"], label="Image")
                out_f = gr.Label(num_top_classes=5, label="Top 5 predictions")
            gr.Button("Classify").click(classify_food, inputs=img_f, outputs=out_f)

demo.launch()
