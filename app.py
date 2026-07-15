import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from dog_training import make_dog_model
from food_training import make_food_model

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")

# ── load models (cached — only runs once) ────────────────────────────────────

@st.cache_resource
def load_models():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.eval()

    dog = make_dog_model()
    dog.load_state_dict(torch.load('model_saver/best_model.pth', map_location='cpu'))
    dog.eval()

    food = make_food_model()
    food.load_state_dict(torch.load('model_saver/best_food.pth', map_location='cpu'))
    food.eval()

    return resnet, dog, food

@st.cache_data
def load_labels():
    with open("imagenet_classes.txt") as f:
        imagenet = [s.strip() for s in f.readlines()]
    with open("dog_labels.txt") as f:
        dogs = [s.strip() for s in f.readlines()]
    with open("food101_labels.txt") as f:
        food = [s.strip() for s in f.readlines()]
    return imagenet, dogs, food

# ── inference ─────────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify(image, model, labels):
    img = preprocess(image.convert("RGB"))
    batch = torch.unsqueeze(img, 0)
    with torch.no_grad():
        output = model(batch)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_ids = torch.topk(probs, 5)
    return [(labels[i.item()], float(p)) for i, p in zip(top_ids, top_probs)]

def show_results(results):
    for label, prob in results:
        col1, col2 = st.columns([4, 1])
        col1.progress(prob, text=label)
        col2.markdown(f"**{prob*100:.1f}%**")

# ── ui ────────────────────────────────────────────────────────────────────────

st.title("Multi-Model Image Classifier")
st.caption("Three ResNet-50 models — general objects, dog breeds, and food.")

resnet_model, dog_model, food_model = load_models()
imagenet_labels, dog_labels, food_labels = load_labels()

tab1, tab2, tab3 = st.tabs(["General (ResNet-50)", "Dog Breeds", "Food"])

with tab1:
    st.caption("Pretrained on ImageNet — recognizes 1,000 everyday objects.")
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="r")
    if f:
        img = Image.open(f)
        st.image(img, width=320)
        with st.spinner("Classifying..."):
            show_results(classify(img, resnet_model, imagenet_labels))

with tab2:
    st.caption("Fine-tuned on 37 dog breeds.")
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="d")
    if f:
        img = Image.open(f)
        st.image(img, width=320)
        with st.spinner("Classifying..."):
            show_results(classify(img, dog_model, dog_labels))

with tab3:
    st.caption("Fine-tuned on 101 food categories.")
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="f")
    if f:
        img = Image.open(f)
        st.image(img, width=320)
        with st.spinner("Classifying..."):
            show_results(classify(img, food_model, food_labels))
