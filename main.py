from data import Ctdataset
from model import BConv
import os
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Loading the data

train_path = os.path.join("files", "test_COVIDx_CT-2A.txt")  
test_path = os.path.join("files", "train_COVIDx_CT-2A.txt")

train_dataset = Ctdataset(train_path)
test_dataset= Ctdataset(test_path)

#initilizing the dataloaders
train_loader = DataLoader(dataset= train_dataset, batch_size= 10, shuffle= True)
test_loader= DataLoader(dataset= test_dataset, batch_size= 10, shuffle= True)
# dataiter= iter(test_loader)
# images, labels = dataiter.next()
model= BConv()


#starting with training
model= BConv().to(device)
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss().cuda()
num_epochs=10
dataiter= iter(train_loader)
image, labels = dataiter.next()
# print(image.shape)
# image= image.reshape(10, 3, 150, 150)
# print(image)
# best_accuracy= 0.0
best_accuracy=0.0
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
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
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        #print(train_accuracy, "after epoch")
        
    train_accuracy=train_accuracy/len(train_loader)
    print(train_accuracy, "train accuracy")
    #train_loss=train_loss/train_count
    
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        print("converting image and labels for testing")
        images= images.float()
        images= images.permute(0,3,1,2)
        images= images.to(device)
        labels= labels.to(device)
        outputs=model(images)
        #images= images.float()
        #_,prediction=torch.max(outputs.data,1)
        #test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/len(test_loader)
    
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
    #Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy=test_accuracy
