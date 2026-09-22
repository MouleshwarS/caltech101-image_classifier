import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

class ConvertToRGB:
    """Converts images to RGB format to handle grayscale/RGBA images in Caltech101."""
    def __call__(self, img):
        return img.convert("RGB")

def create_dataloaders(data_dir: str, batch_size: int = 32):
    """Downloads Caltech101 and creates train/test DataLoaders."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        ConvertToRGB(), 
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    full_dataset = datasets.Caltech101(root=str(data_path), 
                                       transform=train_transforms, 
                                       download=True)

    # Extracting class names: Caltech101 extracts to "101_ObjectCategories" and contains 102 classes (101 + BACKGROUND_Google)
    caltech101_dir = data_path / "caltech101" / "101_ObjectCategories"
    class_names = sorted([entry.name for entry in caltech101_dir.iterdir() if entry.is_dir()])
    
    # 80/20 Train/Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    torch.manual_seed(42)
    train_indices, test_indices = random_split(
        full_dataset, [train_size, test_size]
    )
    train_data = Subset(full_dataset, train_indices.indices)

    # Using a separate dataset instance so that train and test transforms stay independent.
    test_dataset = datasets.Caltech101(
        root=str(data_path), transform=test_transforms, download=False
    )
    test_data = Subset(test_dataset, test_indices.indices)

    num_workers = 4
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True, 
                          persistent_workers=True)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                         num_workers=num_workers, pin_memory=True, 
                         persistent_workers=True)

    return train_loader, test_loader, class_names