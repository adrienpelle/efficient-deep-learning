import torch
import torch.nn as nn
from models_cifar100.resnet import ResNet18
# Assuming the class definitions for BasicBlock, Bottleneck, and ResNet are defined as per your snippet above.



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate a ResNet-18 model
model = ResNet18()

# Count the parameters
total_params = count_parameters(model)

print(f"Total number of parameters in ResNet-18: {total_params}")
