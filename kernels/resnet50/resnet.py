import torch
import torchvision.models as models

# Load ResNet-50
resnet50 = models.resnet50(pretrained=True)

print(resnet50)
