import os
import pandas as pd
path= os.path.join("files", "train_COVIDx_CT-2A.txt")
data= pd.read_csv(path, delimiter=" ")
data= data.values.tolist()
print(data[0] [0])


