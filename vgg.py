import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import Adam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg():
        
    model =  models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 3)
    model.to(device)

    optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss().cuda()
    print(model)
    return model
s= vgg()
