#https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#https://pythonprogramming.net/gpu-deep-learning-neural-network-pytorch/
#https://www.kaggle.com/code/vijaypro/cnn-pytorch-96?scriptVersionId=37837911&cellId=17
#https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# IMPORTING MODULE FILE FROM DIFFERENT FOLDER
import os
import sys
import time

import numpy as np

#sys.path.append(os.getcwd() + "/view")

from backend.models.csvconvert import CSVConvert
from View.ui_main_josefAdditions import Ui_MainWindow
import View.ui_functions

from PyQt5.QtCore import *

class ModelTrainer(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    sendToConsole = pyqtSignal(str)
    saveModelData = pyqtSignal(list)

    def __init__(self, mainUI: Ui_MainWindow):
        super().__init__()
        self.ui = mainUI
        self.generalFunctions = View.ui_functions.GeneralFunctions(self.ui)
        self.epoch = 1
        self.batchSize = 2
        self.model = None
        self.name = "ModelNameHere"
        self.hasFinishedTraining = False
        self.stopTraining = False
        self.testRatio = 0.50
        self.trainRatio = 0.50
        self.transformType = None

        # progress bar
        self.totalProgress = self.epoch * 2
        self.currentProgress = 0

    def setRatio(self, testRatio, trainRatio):
        if (self.hasFinishedTraining):
            self.generalFunctions.sendToConsole("Cannot change ratio has training has finished.")
            return
        
        self.testRatio = float(float(testRatio) / 100.0)
        self.trainRatio = float(float(trainRatio) / 100.0)


    def setEpoch(self, value: int):
        self.epoch = value
        self.totalProgress = value * 2
        print(f"DEBUG CHECKING: Epoch: {value}")
        self.generalFunctions.sendToConsole(f"DEBUG CHECKING: Epoch: {value}")

    def setBatchSize(self, batchSize: int):
        if ((batchSize & (batchSize-1) == 0) and batchSize != 0):
            # base of 2
            print(f"DEBUG CHECKING: Batch size: {batchSize}")
            self.generalFunctions.sendToConsole(f"Batch size: {batchSize} accepted!")
            self.batchSize = batchSize
        else:
            #not base of 2
            self.generalFunctions.sendToConsole(f"Batch size cannot be '{batchSize}' : Expected 2^n integer")
            self.batchSize = 2

    def setModel(self, model):
        self.model = model
    
    def setName(self, name: str):
        self.name = name

    def transformDataset(self, trainCSVFilePath, testCSVFilePath, transformType):

        trainSet, validateSet = self.splitData(trainCSVFilePath, self.trainRatio, self.testRatio)

        #csvconvert wants file path
        self.data = CSVConvert(trainSet.dataset, transformType)
        self.data_val = CSVConvert(validateSet.dataset, transformType)


        # self.data=CSVConvert(os.getcwd() + "/dataset/sign_mnist_train/sign_mnist_train.csv", transform=AlexNetTransform())
        # self.data_val=CSVConvert(os.getcwd() + "/dataset/sign_mnist_test/sign_mnist_test.csv", transform=AlexNetTransform())

    def loadIntoDataLoader(self):
        self.train_loader=DataLoader(dataset=self.data, batch_size=self.batchSize, num_workers=2, shuffle=True)
        self.val_loader=DataLoader(dataset=self.data_val, batch_size=self.batchSize, num_workers=2, shuffle=True)

    def executeModelTrainer(self):
        self.loadIntoDataLoader()
        self.trainAndTest(self.model, self.train_loader, self.val_loader, self.epoch)

    def train(self, trainLoader, model, lossFn, optimizer, device):
        size = len(trainLoader.dataset)

        for batch, (images, labels) in enumerate(trainLoader):
                # Compute prediction and loss
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = lossFn(outputs, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(images)
                    print(f"loss: {loss:>7f}  [{current:>5d}|{size:>5d}]")
                    self.emit_send_to_console(f"loss: {loss:>7f}  [{current:>5d}|{size:>5d}]")

    def test(self, testLoader, model, device):
        total ,correct = 0, 0
        print(f"(Testing)\n-------------------------------")
        self.emit_send_to_console(f"(Testing)\n-------------------------------")
        with torch.no_grad():
            for (images, labels) in testLoader:
                images = images.to(device)
                labels = labels.to(device)
                outputs=model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {(100 * correct / total):>0.1f}%")
        self.emit_send_to_console(f"Test Accuracy: {(100 * correct / total):>0.1f}%")
 
    def trainAndTest(self, model, trainSet, testSet, epoch):
        device = "cpu"
        print(f"Using {device} device")
        self.emit_send_to_console(f"Using {device} device")

        learning_rate = 1e-3
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        model = model.to(device)
        self.currentProgress = 0
        start_time = time.time()
        self.finished.emit(f"Estimated: Begining Training.")
        # Training for set amount of epoch
        for t in range(epoch):
            if (self.stopTraining):
                self.currentProgress = self.currentProgress + 1
                progress = int((self.currentProgress / self.totalProgress) * 100)
                self.progress.emit(progress)

                self.emit_send_to_console(f"Stop training pressed! Skipping to testing.")
            else:
                print(f"(Training)Epoch [{t+1}|{epoch}] \n-------------------------------")
                self.emit_send_to_console(f"(Training)Epoch [{t+1}|{epoch}] \n-------------------------------")
                self.train(trainSet, model, loss_fn, optimizer, device)

                self.currentProgress = self.currentProgress + 1
                progress = int((self.currentProgress / self.totalProgress) * 100)

                tElapsed = time.time() - start_time
                tEstimated = ((tElapsed/self.currentProgress) * (self.totalProgress))
                timeLeft = tEstimated - tElapsed

                self.finished.emit(f"Estimated: {int(timeLeft)} seconds left..")
                self.progress.emit(progress)
                
                print("Done training!")
                self.emit_send_to_console(f"Done training! {self.currentProgress}")
            
            self.test(testSet, model, device)

            self.currentProgress = self.currentProgress + 1 
            progress = int((self.currentProgress / self.totalProgress) * 100)

            tElapsed = time.time() - start_time
            tEstimated = ((tElapsed/self.currentProgress) * (self.totalProgress))
            timeLeft = tEstimated - tElapsed
            self.finished.emit(f"Estimated: {int(timeLeft)} seconds left...")

            self.progress.emit(progress)

            print("Done testing!")
            self.emit_send_to_console(f"Done testing!{self.currentProgress}")
            self.finished.emit("Finished Training!")


        print("Done!")
        self.emit_send_to_console("Done!")
        self.hasFinishedTraining = True

    def saveCurrentModel(self):
        modelFilePath = self.save(self.model, self.name)
        return modelFilePath

    #https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    def save(self, model, name):
        path = name + ".pth"
        model_scripted = torch.jit.script(model) # Export to a TorchScript
        model_scripted.save(path) # Save model via name of model for easy tracing
        print("Saved PyTorch Model State to " + path)
        return path

    #model_name = file name of saved model
    def load(self, model_name):
        device = "cpu"
        path = model_name + ".pth"
        model = torch.jit.load(path, map_location=device)
        return model

    #https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
    # dataset = CSV File
    def splitData(self, dataset, trainRatio, testRatio):

        totalSetSize = len(dataset)
        trainSetSize = int(trainRatio*len(dataset))
        testSetSize = int(testRatio*len(dataset))
        
        print(f"Total amount: {len(dataset)}")
        print(f"Train amount: {int(trainRatio*len(dataset))}")
        print(f"Test amount: {int(testRatio*len(dataset))}")

        if (not ((trainSetSize + testSetSize) == totalSetSize)):
            trainSetSize = trainSetSize + 1
            print(f"Total amount: {len(dataset)}")
            print(f"Train amount: {int(trainRatio*len(dataset))}")
            print(f"Test amount: {int(testRatio*len(dataset))}")

        trainSet, testSet = torch.utils.data.random_split(dataset, [trainSetSize, testSetSize])
        

        # print(trainSet.dataset)
        # print(testSet.dataset)

        # filePath = None
        # # make directory if it does not exist
        # if (not os.path.exists(os.getcwd() + f"/dataset/split_datasets/{self.name}")):
        #     os.makedirs(os.getcwd() + f"/dataset/split_datasets/{self.name}")
        
        # filePath = os.getcwd() + f"/dataset/split_datasets/{self.name}"
        
        # np.savetxt(f"{filePath}/split_trainSet.csv", trainSet.dataset)
        # np.savetxt(f"{filePath}/split_testSet.csv", testSet.dataset)
    
        return trainSet, testSet
    
    def emit_send_to_console(self, text):
        self.sendToConsole.emit(text)
    
    def emit_save_data(self, dataList):
        self.saveModelData.emit(dataList)
    
    def cancelTraining(self):
        self.stopTraining = True
        self.emit_send_to_console("Training stopped! Skipping to testing.")



    # when selected images are taken
    # create new folder where dataset is called dataset_name_selectedimages
    # ImageFolder to that dataset

    # after pressing ok in testing images
    # do below

    # data = torchvision.datasets.ImageFolder(f"{imageFilePath}.jpg"
    # dataloader = DataLoader(dataset=data,batch_size=BATCH_SIZE,num_workers=2,shuffle=True)  
    # model = model you get from loading model



    # index is batch size, if one image index = 0
    # execute showprediction on every button press

    def showPrediction(self, dataloader, idx, model):
        img, label = next(iter(dataloader))
        pred = model(img)
        probs= torch.softmax(pred, dim=1)
        conf= torch.max(probs, 1)
        pred = torch.argmax(pred[idx], dim=0)
        acc = conf[idx] * 100
        # print(f'Fact: {label[idx]}, Prediction: {(torch.argmax(pred[idx], dim=0))}, Accuracy: {conf[idx] * 100}')
        return img[idx], label[idx], pred, acc 

    
    