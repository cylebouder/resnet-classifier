import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from dog_training import make_dog_model
from food_training import make_food_model  # Assuming make_model is defined in training.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# --- Load both classification models ---

# Load the pre-trained ResNet-50 model
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model.eval()

# Load your customhttp://127.0.0.1:6777-trained model
# Load your custom-trained model
custom_model = make_dog_model()
# Make sure your custom model's weights file is in the 'model_saver' directory
custom_model.load_state_dict(torch.load('model_saver/best_model.pth', map_location=torch.device('cpu')))
custom_model.eval()

food_model = make_food_model()
food_model.load_state_dict(torch.load('model_saver/best_food.pth', map_location=torch.device('cpu')))
food_model.eval()



# --- Pre-processing and categories (assumed to be the same for both models) ---

# Pre-processing transformations for the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get the ImageNet class names
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
with open ('dog_labels.txt', 'r') as f:
    dog_categories = [s.strip() for s in f.readlines()] 
with open ('food101_labels.txt', 'r') as f:
    food_categories = [s.strip() for s in f.readlines()] 

# --- Flask routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def classify_resnet():
    """Endpoint to classify an image using the ResNet-50 model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = "uploaded_image_resnet.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        img_preprocessed = preprocess(img)
        batch = torch.unsqueeze(img_preprocessed, 0)
        
        # Classify with the ResNet model
        with torch.no_grad():
            prediction = resnet_model(batch)
        
        # Get the top 5 classification results
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        top_probs, top_catids = torch.topk(probabilities, 5)
        
        # Create a list of dictionaries for the results
        results_list = []
        for prob, catid in zip(top_probs, top_catids):
            results_list.append({
                'label': categories[catid.item()],
                'probability': f"{prob.item()*100:.2f}%"
            })
        
        return jsonify({
            'success': True,
            'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'classifications': results_list
        })

@app.route('/classify_custom', methods=['POST'])
def classify_custom_model():
    """Endpoint to classify an image using the custom-trained model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = "uploaded_image_custom.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        img_preprocessed = preprocess(img)
        batch = torch.unsqueeze(img_preprocessed, 0)
        
        # Classify with the custom model
        with torch.no_grad():
            prediction = custom_model(batch)
        
        # Get the top 5 classification results
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        top_probs, top_catids = torch.topk(probabilities, 5)

        # Create a list of dictionaries for the results
        results_list = []
        for prob, catid in zip(top_probs, top_catids):
            results_list.append({
                'label': dog_categories[catid.item()],
                'probability': f"{prob.item()*100:.2f}%"
            })
        
        return jsonify({
            'success': True,
            'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'classifications': results_list
        })
    
@app.route('/classify_food', methods=['POST'])
def classify_food_model():
    """Endpoint to classify an image using the custom-trained model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = "uploaded_image_food.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        img_preprocessed = preprocess(img)
        batch = torch.unsqueeze(img_preprocessed, 0)
        
        # Classify with the custom model
        with torch.no_grad():
            prediction = food_model(batch)
        
        # Get the top 5 classification results
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        top_probs, top_catids = torch.topk(probabilities, 5)

        # Create a list of dictionaries for the results
        results_list = []
        for prob, catid in zip(top_probs, top_catids):
            results_list.append({
                'label': food_categories[catid.item()],
                'probability': f"{prob.item()*100:.2f}%"
            })
        
        return jsonify({
            'success': True,
            'image_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
            'classifications': results_list
        })
    
if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port="3000", host="0.0.0.0", use_reloader=False)
