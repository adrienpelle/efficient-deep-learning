import torch
import torch.nn as nn
from models_cifar100.resnet import ResNet18_Depthwise  # Adjust this import to match your directory structure.
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint correctly to the specified device
#loaded_cpt = torch.load('trained_student_model.pth', map_location=device)

# Define the model (assuming hyperparameters, if any, are predefined within the model)
model = ResNet18_Depthwise().to(device)

# Load the trained parameters
#model.load_state_dict(loaded_cpt)
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    print(f'Number of trainable parameters: {count_parameters(model)}')
    print(f'Test Accuracy: {test_accuracy:.2f}%, Avg Loss: {np.mean(val_loss):.4f}')

if __name__ == "__main__":
    evaluate(model, testloader, criterion)
