import torch
import argparse
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from models_cifar100.resnet import ResNet18_Depthwise
from models_cifar100.resnet2 import ResNet18
from models_cifar100.resnet3 import ResNet18_Mini



# Argparse for iterations and model loading
parser = argparse.ArgumentParser(description='CIFAR10 Training with PyTorch')
parser.add_argument('--num_epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--model_path', default='', type=str, help='path to trained model (to continue training)')
parser.add_argument('--new_model', action='store_true', help='train a new model or fine-tune an existing one')
args = parser.parse_args()

writer = SummaryWriter()
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

teacher_model = ResNet18().to(device)
student_model = ResNet18_Mini().to(device)

if args.new_model:
    pass  
else:
    student_model.load_state_dict(torch.load(args.model_path))

teacher_model.load_state_dict(torch.load('model_base_best.pth'))

for param in teacher_model.parameters():
    param.requires_grad = False

optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=args.num_epochs)

temperature = 10
alpha = 0.2

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class LossCalculator(nn.Module):
    def __init__(self, temperature, distillation_weight):
        super().__init__()
        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs, targets_a, targets_b, lam):
        # Calculate the KD loss
        soft_target_loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1),
                                      F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.temperature ** 2)

        # Calculate the Mixup loss
        hard_target_loss = lam * F.cross_entropy(outputs, targets_a) + (1 - lam) * F.cross_entropy(outputs, targets_b)

        # Combine losses
        total_loss = (soft_target_loss * self.distillation_weight) + (hard_target_loss * (1 - self.distillation_weight))
        return total_loss


loss_calculator = LossCalculator(temperature, alpha).to(device)
best_val_accuracy=0
early_stopping_patience=20

for epoch in range(args.num_epochs):
    student_model.train()
    teacher_model.eval()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    student_model.train()
    teacher_model.eval()
    for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha)
        
        optimizer.zero_grad()
        student_outputs = student_model(inputs)
        teacher_outputs = teacher_model(inputs)
        
        loss = loss_calculator(student_outputs, labels, teacher_outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(student_outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}: Training Accuracy: {train_accuracy}%")
    writer.add_scalar('Training/Loss', running_loss / len(trainloader), epoch)
    writer.add_scalar('Training/Accuracy', train_accuracy, epoch)
    
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
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100 * correct_val / total_val
    print(f"Validation Accuracy: {val_accuracy}%")
    writer.add_scalar('Validation/Loss', val_loss / len(testloader), epoch)
    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        torch.save(student_model.state_dict(), 'finalv4.pth')
        print('New best model saved')
    else:
        early_stopping_counter += 1
        print(early_stopping_counter)
    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break


    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', lr, epoch)

    
    scheduler.step(val_accuracy)
    

    
    for name, param in student_model.named_parameters():
        writer.add_histogram(f'{name}/weights', param, epoch)
        writer.add_histogram(f'{name}/gradients', param.grad, epoch)

writer.close()
print('Finished Training')
