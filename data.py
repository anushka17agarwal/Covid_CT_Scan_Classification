import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import pandas as pd

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
        img= np.array(Image.open(img_path))
        img= img.astype(float)
        label= self.labels[idx]
        #label= self.class_map[label]
        return  label
    
        

train_path = os.path.join("files", "test_COVIDx_CT-2A.txt")  
test_path = os.path.join("files", "train_COVIDx_CT-2A.txt")

dataset = Ctdataset(train_path)
train_loader = DataLoader(dataset= dataset, batch_size= 3, shuffle= True)
for i in train_loader:
    print(i)

