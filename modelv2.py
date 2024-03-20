import torch
import argparse
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models_cifar100.resnet import ResNet18_Depthwise
from models_cifar100.resnet2 import ResNet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
  # Ensure this path is correct



writer = SummaryWriter()
# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load models
teacher_model = ResNet18().to(device)
student_model = ResNet18_Depthwise().to(device)
# Load pre-trained models here
teacher_model.load_state_dict(torch.load('model_base_best.pth'))
# Optionally, load pre-trained student model
student_model.load_state_dict(torch.load('test_resnet_best2.pth'))

# Make teacher model not trainable
for param in teacher_model.parameters():
    param.requires_grad = False

optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

# Temperature for distillation and alpha for balancing the loss
temperature = 4
alpha = 0.7

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

def distillation_loss_with_mixup(y_student, y_teacher, y_a, y_b, lam, T, alpha):
    KL_loss = F.kl_div(F.log_softmax(y_student / T, dim=1),
                       F.softmax(y_teacher / T, dim=1),
                       reduction='batchmean') * (T * T * alpha)
    true_label_loss = lam * criterion(y_student, y_a) + (1 - lam) * criterion(y_student, y_b)
    return KL_loss + true_label_loss

# Training and validation loop
num_epochs = 30

for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()  
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
        
        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        teacher_outputs = teacher_model(inputs)
        
        loss = distillation_loss_with_mixup(student_outputs, teacher_outputs, targets_a, targets_b, lam, temperature, alpha)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(student_outputs, 1)
        total_train += labels.size(0)
        correct_train += ((predicted == labels).float() * lam + (predicted == labels.flip(0)).float() * (1 - lam)).sum().item()
    
    training_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}: Loss: {running_loss / len(trainloader)}, Training Accuracy: {training_accuracy}%")

    # Validation phase
    student_model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Loss: {val_loss / len(testloader)}, Validation Accuracy: {val_accuracy}%")

    # Logging Train Accuracy
    writer.add_scalar('Train/Accuracy', 100. * correct_train / total_train)

    # Logging Validation Loss
    writer.add_scalar('Validation/Loss', np.mean(val_loss))

    # Logging Validation Accuracy
    writer.add_scalar('Validation/Accuracy', val_accuracy)

    for name, param in student_model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

writer.close()
# Save the trained student model
torch.save(student_model.state_dict(), 'trained_student_model.pth')
print('Finished Training')
