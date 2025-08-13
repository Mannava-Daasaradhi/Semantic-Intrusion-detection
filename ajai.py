import os
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # Import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("✅ GPU is available.")
else:
    print("ℹ️ No GPU found, running on CPU.")

# Data loading and preprocessing
from pathlib import Path

# Base directory (update if needed)
base_dir = Path(r"D:\Advance Data Structures ML project\diabetic-retinopathy-detection")

# Define paths for train and test subfolders
train_dirs = [base_dir / f"train_{i:03d}" / "train" / "train" for i in [1, 2, 3]]
labels_path = base_dir / "trainLabels" / "trainLabels" / "trainLabels.csv"

# Read labels CSV
labels_df = pd.read_csv(labels_path)
print("Label distribution:\n", labels_df['level'].value_counts())

# Collect all training image file paths
train_image_paths = []
for d in train_dirs:
    train_image_paths.extend([str(p) for p in d.glob("*.*")])
print(f"Found {len(train_image_paths)} training images.")

# Build dataframe with image paths and corresponding labels
df = pd.DataFrame({'image_path': train_image_paths})
df['image_id'] = df['image_path'].apply(lambda x: Path(x).stem)
df = df.merge(labels_df, left_on='image_id', right_on='image')
df = df[['image_path', 'level']].rename(columns={'level': 'label'})
df['label'] = df['label'].astype(int)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# PyTorch Dataset
class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path)
        label = self.dataframe.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = RetinopathyDataset(train_df, transform=transform)
val_dataset = RetinopathyDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Add tqdm to the training loader
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Convert labels to long
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=running_loss/len(progress_bar))

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))