import torch
import torch.nn as nn
from models_cifar100.resnet import ResNet18



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = ResNet18()

total_params = count_parameters(model)

print(f"Total number of parameters in ResNet-18: {total_params}")
