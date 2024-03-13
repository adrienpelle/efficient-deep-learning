import torch
import argparse
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from models_cifar100.resnet import ResNet18
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Model Training on CIFAR10')
parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train for')
args = parser.parse_args()

# TensorBoard Writer
writer = SummaryWriter()

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and mixing lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Data preprocessing and loading
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

# Model setup
model = ResNet18().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1)

# Training loop
n_epochs = args.num_epochs
best_accuracy = 0
early_stopping_counter = 0
early_stopping_patience = 20

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    epoch_loss = []
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch+1}/{n_epochs}')
    for i, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (lam * (predicted == targets_a).sum().float() + (1 - lam) * (predicted == targets_b).sum().float()).item()

    # Validation
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
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    validation_accuracy = 100 * correct_test / total_test

    # Print epoch results
    print(f"Epoch {epoch+1}/{n_epochs}:")
    print(f"  Train Loss: {running_loss / len(trainloader):.4f} | Train Accuracy: {100. * correct_train / total_train:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    
    # Checkpointing and early stopping
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'test_resnet_best2.pth')
    else:
        early_stopping_counter += 1
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break
    

    current_lr = scheduler.optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)
    # Logging Train Loss
    writer.add_scalar('Train/Loss', np.mean(epoch_loss))

    # Logging Train Accuracy
    writer.add_scalar('Train/Accuracy', 100. * correct_train / total_train)

    # Logging Validation Loss
    writer.add_scalar('Validation/Loss', np.mean(val_loss))

    # Logging Validation Accuracy
    writer.add_scalar('Validation/Accuracy', validation_accuracy)

    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

torch.save(model.state_dict(), 'end_of_training_model.pth')
writer.close()
print('Finished Training')
