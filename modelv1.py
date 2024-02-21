import torch
import argparse
from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import models_cifar100
from models_cifar100.resnet import ResNet18
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm
from tqdm import tqdm

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser=argparse.ArgumentParser(prog='Model Training on CIFAR10')
parser.add_argument('num_epochs')
args=parser.parse_args()

# TensorBoard Writer
writer = SummaryWriter()

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, mixed targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Data preprocessing and loading
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])
rootdir = './data/cifar10'
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
testloader = DataLoader(c10test, batch_size=32)
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)

# Model setup
model = ResNet18().to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

total_all_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters (including non-trainable): {total_all_params}")
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1, verbose=True)

# Training parameters
n_epochs = int(args.num_epochs)
best_accuracy = 0
early_stopping_counter = 0
early_stopping_patience = 5

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    epoch_loss = []
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch+1}/{n_epochs}')
    for i, data in progress_bar:
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # Apply mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Mixup criterion
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        # Adjust accuracy calculation for mixup
        correct_train += (lam * (predicted == targets_a).sum().float() + (1 - lam) * (predicted == targets_b).sum().float()).item()
    
    # Log training metrics
    train_accuracy = 100 * correct_train / total_train
    
    writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    #Add Weights Gradients to Tensorboard
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), epoch)

    # Validation phase
    val_loss = []
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    # Log validation metrics
    test_accuracy = 100 * correct_test / total_test
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        early_stopping_counter = 0  
        checkpoint_path = 'model_best.pth'
        torch.save(model.state_dict(), checkpoint_path)
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        break
    writer.add_scalar('Loss/validation', np.mean(val_loss), epoch)
    writer.add_scalar('Accuracy/validation', test_accuracy, epoch)
    
    # Optionally log learning rate
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch)
    
    # Early stopping and model checkpointing remains unchanged

    print(f'Epoch {epoch+1}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {test_accuracy:.2f}%')


torch.save(model.state_dict(), 'end_of_training_model.pth')
print('Finished Training')
# Cleanup and saving model
writer.close()
