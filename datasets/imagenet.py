import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os

class CustomImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_dir in self.root.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob('*.JPEG'):
                    self.samples.append((str(img_path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Open image using PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)  # Direct transform, not with keyword args
            
        return image, label 