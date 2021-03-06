import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import Adam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def resnet():
        
    model =  models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.to(device)

    optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss().cuda()
    return model