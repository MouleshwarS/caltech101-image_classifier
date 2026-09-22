import torch.nn as nn
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights
)

class CustomCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 102):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))

def create_vgg16(num_classes: int = 102) -> nn.Module:
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False # Freeze backbone
        
    classifier = model.classifier[6]
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected VGG16 classifier layer 6 to be nn.Linear")
    in_features = classifier.in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )
    return model

def create_resnet50(num_classes: int = 102) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False # Freeze backbone
        
    if not isinstance(model.fc, nn.Linear):
        raise TypeError("Expected ResNet50 fc layer to be nn.Linear")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )
    return model

def create_efficientnet(num_classes: int = 102) -> nn.Module:
    """EfficientNet-V2-Small: Highly optimized for parameter efficiency."""
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
    # EfficientNet classifier is a Sequential where index 1 is the Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )
    return model

def create_convnext(num_classes: int = 102) -> nn.Module:
    """ConvNeXt-Tiny: provides Transformer-level accuracy while keeping the simplicity and speed of a pure CNN."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
        
    # Classifier is a Sequential where index 2 is the Linear layer
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3), nn.Linear(in_features, num_classes)
    )
    return model