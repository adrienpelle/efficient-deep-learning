import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from binarryconnect import BC
from torch.utils.data import DataLoader
from models_cifar100.resnet import ResNet18  # Adjust import path as needed
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

# Setup argparse
parser = argparse.ArgumentParser(description='Train and/or Validate ResNet18 on CIFAR10 with BinaryConnect')
parser.add_argument('--mode', type=str, choices=['both', 'validate'], default='both',
                    help='Choose operation mode: "both" for training and validation, "validate" for validation only.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for.')
parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save or load the model.')
args = parser.parse_args()

writer = SummaryWriter()

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalization for validation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path))
bc = BC(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_and_validate(model, bc, train_loader, test_loader, criterion, optimizer, epochs=20, mode='both'):
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        if mode == 'both':
            model.train()
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                bc.binarization()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                bc.restore()
                optimizer.step()
                bc.clip()
                
                # TensorBoard logging for training loss
                writer.add_scalar('Loss/train', loss.item(), epoch)

            # Log weights and gradients
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}/weights', param, epoch)
                writer.add_histogram(f'{name}/grads', param.grad, epoch)
        
        # Validation step
        val_acc = validate(model, bc, test_loader, criterion)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)

    return best_epoch, best_acc

def validate(model, bc, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            bc.binarization()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    bc.restore()

if __name__ == "__main__":
    if args.mode == 'both':
        print("Starting training and validation...")
        best_epoch, best_acc = train_and_validate(model, bc, train_loader, test_loader, criterion, optimizer, epochs=args.epochs, mode='both')
        print(f'Best Model: Epoch {best_epoch+1}, Validation Accuracy: {best_acc:.2f}%')
    elif args.mode == 'validate':
        print("Starting validation only...")
        if os.path.isfile(args.model_path):
            model.load_state_dict(torch.load(args.model_path))
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"No model found at {args.model_path}. Please check the path or train a model first.")
            exit()
        validate_accuracy = validate(model, bc, test_loader, criterion)
        print(f'Validation Accuracy: {validate_accuracy:.2f}%')
    writer.close()
