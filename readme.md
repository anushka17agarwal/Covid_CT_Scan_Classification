# Covid CT Classification
This is the implementation of a convolutional Neural Network on <a href= "https://www.kaggle.com/hgunraj/covidxct" >COVIDx CT  </a>

## About the Dataset

- The following dataset contains CT images from 194,922 CT slices from 3,745 patients and 201,103 CT slices from 4501 patients respectively.

- Classes are zero-indexed with Normal=0, Pneumonia=1, and COVID-19=2
- The COVIDx CT-2 dataset is released as a directory of images (2A_images) and associated label files ({train,val,test}_COVIDx_CT-2A.txt) indicating classes and bounding boxes for the body region.


Link to download the dataset: <a href="https://www.kaggle.com/hgunraj/covidxct"> Here </a>


## Downloading the trained Model:
<a href="https://drive.google.com/file/d/1n-7OYvLqkg0E5VNBypnV5tDTAFSlH_o_/view?usp=sharing"> Here </a>


## Accuracy Scores

## After 3 epochs
### basic cnn
- Train Loss: 0.026 | Accuracy: 99.146
- Test Loss: 1.190 | Accuracy: 76.404

###    resnet

- Train Loss: 0.024 | Accuracy: 99.229
- Test Loss: 1.056 | Accuracy: 79.012
 

### VGG-16
- Train Loss: 13.014 | Accuracy: 70.982
- Test Loss: 12.807 | Accuracy: 58.678