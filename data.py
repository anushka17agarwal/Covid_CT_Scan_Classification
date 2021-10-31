import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as fn

class Ctdataset(Dataset):
    def __init__(self, path):

        self.data= pd.read_csv(path, delimiter=" ")
        data= self.data.values.tolist()
        self.image= []
        self.labels=[]
        for i in data:
            self.image.append(i[0])
            self.labels.append(i[1])

        print(len(self.image), len(self.labels))
        #self.class_map = {"0": 0, "1":1 , "2": 2}  

    def __len__(self):
        return len(self.image)
        

    def __getitem__(self, idx):
        img_path = os.path.join("2A_images", self.image[idx])
        img= Image.open(img_path).convert("RGB")
        img= img.resize((150, 150))
        img= np.array(img)
        img= img.astype(float)
        #img = fn.resize(img, size=[150, 150])
        #img= img.resize(150, 150)
        label= self.labels[idx]
        # transform = A.Compose(
        # [
        #     A.Resize(height=250, width=250),
        #     ToTensorV2()
        # ]
        # )
        # if self.transform is not None:
        #     augumentations = self.transform(image= img)
        #     image= augumentations["img"]

        #label= self.class_map[label]
        return img, label
    
        

train_path = os.path.join("files", "test_COVIDx_CT-2A.txt")  
test_path = os.path.join("files", "train_COVIDx_CT-2A.txt")

dataset = Ctdataset(train_path)
train_loader = DataLoader(dataset= dataset, batch_size= 10, shuffle= True)

dataiter= iter(train_loader)
images, labels = dataiter.next()
print(len(images))

