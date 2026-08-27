import torch
import torch.nn as nn
from torchvision import models

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture.lower() == "resnet18":
        model = models.resnet18(weights=None)
        
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        return model
    else:
        raise ValueError(f"Architecture {architecture} is not supported in this pipeline.")
