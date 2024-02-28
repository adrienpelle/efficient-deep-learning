import torch
import torch.nn as nn
import numpy as np

class BC():
    def __init__(self, model):
        self.model = model
        self.saved_params = []
        self.target_modules = []
        self.bin_range = []

        # Count Conv2d and Linear modules to determine binarization targets
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m)

        # Initialize bin_range for indexing
        self.bin_range = list(range(len(self.target_modules)))

    def save_params(self):
        for index, module in enumerate(self.target_modules):
            self.saved_params[index].copy_(module.weight.data)

    def binarization(self):
        self.save_params()  # Save current full precision parameters

        # Binarize weights: Iterate through target modules and binarize
        for module in self.target_modules:
            # Use sign function for binarization
            weight = module.weight.data
            module.weight.data = weight.sign()

    def restore(self):
        # Restore the saved full-precision weights
        for index, module in enumerate(self.target_modules):
            module.weight.data.copy_(self.saved_params[index])

    def clip(self):
        # Clip weights to [-1, 1] using Hardtanh
        for module in self.target_modules:
            module.weight.data = nn.functional.hardtanh_(module.weight.data)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)