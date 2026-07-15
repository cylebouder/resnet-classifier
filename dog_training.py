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



def make_dog_model():
    # Load pre-trained ResNet-50 model from torchvision
    model = models.resnet50(weights="IMAGENET1K_V1") 
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 37 # or however many
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

        classes = set()
        for fname in os.listdir(root_dir):
            if fname.strip():
                class_label = fname.strip().rsplit('_',1)[0].lower()
                classes.add(class_label)

        self.id_to_class = dict(enumerate(classes))
        self.class_to_id = {cls : indent for (indent, cls) in self.id_to_class.items()}

        self.data_items = []
        for fname in os.listdir(root_dir):
            if fname.strip().endswith('.jpg'):
                class_label = fname.strip().rsplit('_',1)[0].lower()
                #self.data_items.append((fname, self.class_to_id[class_label] ))
                
                self.data_items.append((os.path.join(self.root_dir, fname), self.class_to_id[class_label]))

    def __len__(self):

        return len(self.data_items)


    def __getitem__(self, idx):
        
        # 1. Get (path, label) for this index
        path, label = self.data_items[idx]
        # 2. Load the image with PIL.Image.open(path).convert("RGB")
        image = Image.open(path).convert("RGB")
        # 3. Apply transform if provided
        if self.transform:
            image = self.transform(image)
            #self.data_items.append((fname, self.class_to_id[cls]))
        # 4. Return (image_tensor, label)
        return image, label
    
    def save_labels(self, filename):
        with open(filename, 'w') as f:
            for idx, cls in self.id_to_class.items():
                f.write(f"{cls}\n")   
            




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
    model_path = os.path.join(save_dir, 'best_model.pth')
    
    best_val_acc = 0.0  # Initialize variable to track the best validation accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
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
    
    

    #model = make_model()
    dogData = CustomDataset("dogImages", transform=preprocess)
    dogData.save_labels('dog_labels.txt')
    
    
    print(len(dogData))
    foodData = CustomDataset("food-101", transform=preprocess)

    #train_dataset, test_dataset = random_split(dogData, [0.8, 0.2])
    train_dataset, test_dataset = random_split(dogData, [0.8, 0.2])

    test_loader = DataLoader(test_dataset, batch_size = 32 , shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=False)
    
    print(len(dogData))
    
    train_model(make_model(), train_loader, test_loader, 30)
    