import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os

class CustomImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Get class folders
        self.classes = sorted([d for d in os.listdir(self.root / split) if d.startswith('n')])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / split / class_name
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    self.samples.append((
                        str(class_dir / img_name),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 