import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from models_cifar100.resnet import ResNet18  # Adjust this import to match your directory structure.
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load checkpoint correctly to the specified device
loaded_cpt = torch.load('model_best.pth', map_location=device)

# Define the model
model = ResNet18().to(device)

# Load the trained parameters
model.load_state_dict(loaded_cpt)
model.half()  # Convert model to half precision

# Define normalization and transformations for the test dataset
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

# Load CIFAR-10 test dataset
rootdir = './data/cifar10'
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
testloader = DataLoader(c10test, batch_size=32, shuffle=False)

# Define the loss function
criterion = nn.CrossEntropyLoss()

def evaluate(model, testloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device).half(), labels.to(device)  # Convert inputs to half precision, labels remain the same
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # Calculate and print the test accuracy
    test_accuracy = 100 * correct_test / total_test
    return test_accuracy, np.mean(val_loss)

# Pruning function
def prune_network(model, pruning_rate):
    for name, module in model.named_modules():
        # Skip modules that do not have 'weight' attribute
        if not hasattr(module, 'weight'): continue
        prune.l1_unstructured(module, name='weight', amount=pruning_rate)
        prune.remove(module, 'weight')  # Make the pruning permanent

# Main
if __name__ == "__main__":
    accuracies = []
    pruning_rates = np.linspace(0, 0.9, 10)  # From 0% to 90% pruning

    for pruning_rate in pruning_rates:
        # Prune the network
        prune_network(model, pruning_rate)

        # Evaluate the pruned network
        accuracy, _ = evaluate(model, testloader, criterion)
        accuracies.append(accuracy)
        print(f"Pruning Rate: {pruning_rate*100:.1f}%, Test Accuracy: {accuracy:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_rates*100, accuracies, marker='o')
    plt.title('Test Accuracy vs. Pruning Percentage')
    plt.xlabel('Pruning Percentage')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.show()
