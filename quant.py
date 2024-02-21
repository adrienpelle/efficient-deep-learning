import torch
import torch.nn as nn
import models_cifar100
from models_cifar100.resnet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We load the dictionnary
loaded_cpt = torch.load('model_best.pth')
# Fetch the hyperparam valu

hparam_bestvalue = loaded_cpt['hyperparam']

# Define the model 
model = ResNet18(hyperparam = hparam_bestvalue).to(device)

# Finally we can load the state_dict in order to load the trained parameters 
model.load_state_dict(loaded_cpt['net'])

# If you use this model for inference (= no further training), you need to set it into eval mode
model.eval()
