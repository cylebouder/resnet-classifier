from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def make_food_model():
    # Load pre-trained ResNet-50 model from torchvision
    model = models.resnet50(weights="IMAGENET1K_V1") 
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 101 # or however many
    model.fc = nn.Linear(2048, num_classes) 

    for param in model.fc.parameters(): # unfreeze new model parameters
        param.requires_grad = True
    return model


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # List all class subdirectories (e.g., apple_pie, baby_back_ribs, etc.)
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_id = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.id_to_class = {idx: cls_name for cls_name, idx in self.class_to_id.items()}

        # Collect all (path, label) pairs
        self.data_items = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data_items.append((os.path.join(cls_folder, fname), self.class_to_id[cls_name]))

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        path, label = self.data_items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def save_labels(self, filename):
        with open(filename, 'w') as f:
            for idx, cls in self.id_to_class.items():
                f.write(f"{idx}: {cls}\n")

            




def evaluate(model, loader, loss, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss_val = loss(outputs, labels)
            val_loss += loss_val.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_loss = val_loss / val_total
    accuracy = val_correct / val_total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, num_epochs, learning_rate=0.001, device='mps'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    save_dir = 'model_saver'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, 'best_food.pth')
    
    best_val_acc = 0.0  # Initialize variable to track the best validation accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")




if __name__ == "__main__":
    data_root = "/Users/dylanvatanapradit/projct rest net/food-101/images"
    
    foodData = CustomDataset(data_root, transform=preprocess)
    foodData.save_labels("food101_labels.txt")

    train_dataset, test_dataset = random_split(foodData, [int(0.8*len(foodData)), len(foodData) - int(0.8*len(foodData))])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Total images: {len(foodData)}")
    print(f"Classes: {len(foodData.classes)}")

    model = make_model()
    train_model(model, train_loader, test_loader, num_epochs=30, device="mps")

    