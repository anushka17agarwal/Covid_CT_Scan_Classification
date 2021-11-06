import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.transforms.functional as TF
import torch.nn as nn

class BConv(nn.Module):
    def __init__(self, out=3):
        super(BConv, self).__init__()
        #(10, 150, 150, 3)
        self.conv1= nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        # self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        # self.bn3=nn.BatchNorm2d(num_features=32)
        # self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features= 20*75*75, out_features=3)
        
    def forward(self,input):
        output=self.conv1(input)
        #print("output 1", output.shape)
        output=self.bn1(output)
        #print("output 1", output.shape)
        output=self.relu1(output)
        #print("output 1", output.shape)
        output=self.pool(output)
        #print("output 1", output.shape)
        output=self.conv2(output)
        #print("output 1", output.shape)
        output=self.relu2(output)
        #print("output 1", output.shape)
        # output=self.conv3(output)
        # output=self.bn3(output)
        # output=self.relu3(output)
        #print(output.shape)
            
            #Above output will be in matrix form, with shape (256,32,75,75)
        
        output=output.contiguous().view(-1,20*75*75)
        
            
        output=self.fc(output)
            
        return output

