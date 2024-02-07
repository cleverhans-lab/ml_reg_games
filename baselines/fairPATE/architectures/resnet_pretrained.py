import torch
import torchvision
import torch.nn as nn
from torchvision.models import ResNet50_Weights

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def pretrained_resnet50():
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(2048, 2)
    model.fc.train(True)
    return model


if __name__ == "__main__":
    pretrained_resnet50()