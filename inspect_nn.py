import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import re
from models_cifar100.resnet import ResNet18

# Assuming these are the filenames for your models
model_files = [
    'pruned_99.000percent_89.11accuracy.pth',
    'pruned_99.225percent_88.29accuracy.pth',
    'pruned_99.450percent_87.79accuracy.pth',
    'pruned_99.675percent_85.91accuracy.pth',
    'pruned_99.900percent_82.53accuracy.pth'
]

# Extract accuracies from filenames using regex
accuracies = []
for name in model_files:
    match = re.search(r'(\d+\.\d+)accuracy', name)
    if match:
        accuracy = float(match.group(1))
        accuracies.append(accuracy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_counts = []

# Iterate over each model file to count non-zero parameters
for model_file in model_files:
    # Make sure to specify the correct path to your model files
    model_path = f"D:/Mathis/Documents/efficient-deep-learning/{model_file}"  # Update this to the correct path
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Count non-zero parameters
    total_non_zero = sum(torch.count_nonzero(param).item() for _, param in model.named_parameters())
    param_counts.append(total_non_zero)

# Plotting the accuracies vs number of non-zero parameters
plt.figure(figsize=(10, 6))
plt.scatter(param_counts, accuracies, color='blue')
plt.plot(param_counts, accuracies, linestyle='-', color='red')  # Connect points with a line
plt.title('Accuracy vs. Number of Non-Zero Parameters')
plt.xlabel('Number of Non-Zero Parameters')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()
