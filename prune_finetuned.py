import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models_cifar100.resnet import ResNet18_Depthwise
import argparse
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(description='Model Performance Evaluation and Fine-tuning')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning the best model')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, testloader, criterion):
    model.eval()
    val_loss = []
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f'Number of trainable parameters: {count_parameters(model)}')
    print(f'Test Accuracy: {test_accuracy:.2f}%, Avg Loss: {np.mean(val_loss):.4f}')
    return test_accuracy, np.mean(val_loss)

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([transforms.ToTensor(), normalize])
rootdir = './data/cifar10'
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
testloader = DataLoader(c10test, batch_size=32)
criterion = nn.CrossEntropyLoss()

# Load models and evaluate them to find the best one
models = []
best_accuracy = 0
for x in range(2, 10):
    model = ResNet18_Depthwise().to(device)
    model_path = f'best_distillated_model{x}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    accuracy, _ = evaluate(model, testloader, criterion)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best Model Test Accuracy: {best_accuracy:.2f}%")


optimizer = optim.Adam(best_model.parameters(), lr=0.0001)

def prune_and_finetune(model, fine_tune_epochs, train_loader, testloader, optimizer, criterion):
    # Iterate over pruning percentages from 10% to 50%
    for pruning_percentage in np.linspace(0.1, 0.5, 5):
        current_pruning = int(pruning_percentage * 100)  # Convert to an integer percentage for naming
        print(f"\nPruning with {current_pruning}% sparsity")

        # Prune model globally across all Conv2d and Linear layers
        parameters_to_prune = [
            (module, 'weight') for module in model.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]
        
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=pruning_percentage)
        
        # Remove the pruning reparameterization for a cleaner model
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        best_val_accuracy = 0
        save_path = f'{current_pruning}_pruned.pth'  # Dynamic save path based on pruning percentage

        model.train()
        for epoch in range(fine_tune_epochs):
            train_loss = []
            model.train()  # Ensure model is in training mode
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            
            # Validation phase
            val_accuracy, val_loss = evaluate(model, testloader, criterion)
            print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            
            # Save the model if it has the best validation accuracy so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), save_path)
                print(f"Saved improved model checkpoint to {save_path} with {val_accuracy:.2f}% accuracy.")

        print(f"Finished pruning at {current_pruning}% sparsity. Best Validation Accuracy: {best_val_accuracy:.2f}%")


# Example usage of the modified function
prune_and_finetune(best_model, args.epochs, trainloader, testloader, optimizer, criterion)
