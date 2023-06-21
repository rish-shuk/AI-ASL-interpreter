
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


#The following csvconvert custom class was sourced by:
# https://www.kaggle.com/code/vijaypro/cnn-pytorch-96?scriptVersionId=37837911&cellId=10
# https://www.kaggle.com/code/wacholder000/simple-convolution-nn-in-pytorch-test-acc-95
class CSVConvert(Dataset):
    def __init__(self,csv, transform):
        #read the csv file
        self.csv_df= pd.read_csv(csv)
        #load transform dependent on type of architecture model
        self.transform = transform
        
    def __getitem__(self,index):

        #get the image and label
        img = self.csv_df.iloc[index,1:].values.reshape(28,28)
        label = self.csv_df.iloc[index,0]

        img = torch.Tensor(img).unsqueeze(0)

        #Apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return self.csv_df.shape[0]

# function to show an image with its corresponding label
def show_img(img, label):
    img = img.squeeze()
    img = img * 40. + 159.

    # if image has 3 channels, modify the image to only have 2 dimensions
    if len(img.shape)==3:
        img = img[0,:,:]

    #remove grid
    plt.axis("off")
    plt.imshow(img, interpolation='bicubic')
    print(label)

# Following tranforms are for lenet5 model in order to load a 32x32 image with 1 channel
def LenetTransform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform

def Img_LenetTransform():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform

# Following tranforms are for cnn model in order to load a 224x224 image with 1 channel
def CNNTransform():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0), (1))
        ])
    return transform

def Img_CNNTransform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0), (1))
        ])
    return transform

#Source of model transform for alexnet: https://blog.paperspace.com/alexnet-pytorch/
def AlexNetTransform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),

        #https://stackoverflow.com/a/74730477
        transforms.Grayscale(num_output_channels=3), #AlexNet needs 3 channels to train/test
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.3,), std=(0.5,))
    ])
    return transform

def Img_AlexNetTransform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),

        #https://stackoverflow.com/a/74730477
        transforms.Grayscale(num_output_channels=3), #AlexNet needs 3 channels to train/test
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.3,), std=(0.5,))
    ])
    return transform


