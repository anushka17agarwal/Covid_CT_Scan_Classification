from data import Ctdataset
from model import BConv
import os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

#for resnet
from resnet import resnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Loading the data

train_path = os.path.join("files", "train_COVIDx_CT-2A.txt")  
test_path = os.path.join("files", "test_COVIDx_CT-2A.txt")

train_dataset = Ctdataset(train_path)
test_dataset= Ctdataset(test_path)

#initilizing the dataloaders
train_loader = DataLoader(dataset= train_dataset, batch_size= 10, shuffle= True)
test_loader= DataLoader(dataset= test_dataset, batch_size= 10, shuffle= True)
# dataiter= iter(test_loader)
# images, labels = dataiter.next()
model= resnet()
#print(len(train_dataset))
#print(len(test_dataset))
#starting with training
model= BConv().to(device)
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss().cuda()
num_epochs=13
# dataiter= iter(train_loader)
# image, labels = dataiter.next()
# print(image.shape)
# image= image.reshape(10, 3, 150, 150)
# print(image)
# best_accuracy= 0.0
best_accuracy=0.0
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss= 0
    correct = 0
    total= 0
    train_accu= []
    train_losses= []
    for i, (images,labels) in enumerate(train_loader):
        images= images.permute(0,3,1,2)
        images= images.float()
        #print("converting image and labels for training")
        images= images.to(device)
        labels= labels.to(device)
        
        #print(labels)    
        #labels= labels.flatten()
        optimizer.zero_grad()
        #images= images.float()
        outputs=model(images)
        #print("training model")
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        #print(images.size(0), images.size(1),images.size(2),"images.size(0)")
        running_loss += loss.item()
        _,predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss=running_loss/len(train_loader)
    accu=100.*correct/total
    accu=100.*correct/total
    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))


    
      
    model.eval()
    
    eval_losses=[]
    eval_accu=[]
    running_loss=0
    correct=0
    total=0

    for i, (images,labels) in enumerate(test_loader):
        images= images.permute(0,3,1,2)
        images= images.float()
        #print("converting image and labels for training")
        images= images.to(device)
        labels= labels.to(device)
        outputs=model(images)
 
        loss= loss_function(outputs,labels)
        running_loss+=loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
            
    test_loss=running_loss/len(test_loader)
    accu=100.*correct/total
        
    eval_losses.append(test_loss)
    eval_accu.append(accu)
        
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 




    
    
    torch.save(model.state_dict(),'best_checkpoint.model')
  