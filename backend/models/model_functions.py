
# The following code is sourced and inspired from these following sources:
#https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#https://pythonprogramming.net/gpu-deep-learning-neural-network-pytorch/
#https://www.kaggle.com/code/vijaypro/cnn-pytorch-96?scriptVersionId=37837911&cellId=17
#https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

import torch
import torch.nn as nn

def train(trainLoader, model, lossFn, optimizer, device):
    size = len(trainLoader.dataset)

    # Training loop
    for batch, (images, labels) in enumerate(trainLoader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = lossFn(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"loss: {loss:>7f}  [{current:>5d}|{size:>5d}]")


def test(testLoader, model, device):

    total ,correct = 0, 0
    print(f"(Testing)\n-------------------------------")
    # Turn off gradients for validation, saves memory
    with torch.no_grad():
        for (images, labels) in testLoader:
            images = images.to(device)
            labels = labels.to(device)

            outputs=model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print accuracy
    print(f"Test Accuracy: {(100 * correct / total):>0.1f}%")
    print("Done Testing!")


def trainAndTest(model, trainSet, testSet, epoch):

    # Use GPU if available
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # Training for set amount of epoch
    for t in range(epoch):
        print(f"(Training)Epoch [{t+1}|{epoch}] \n-------------------------------")
        train(trainSet, model, loss_fn, optimizer, device)
        print("Done training!")
        test(testSet, model, device)
    print("Done!")

#https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
def save(model, name):
    path = name + ".pth"
    model_scripted = torch.jit.script(model) # Export to a TorchScript
    model_scripted.save(path) # Save model via name of model for easy tracing
    print("Saved PyTorch Model State to " + path)

def load(model_name, device):
    if device is None:
        device = 'cpu'
    path = model_name + ".pth"
    model = torch.jit.load(path, map_location=device)
    return model

#https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
def splitData(dataset, trainRatio, testRatio):
    trainSet, testSet = torch.utils.data.random_split(dataset, [int(trainRatio*len(dataset)), int(testRatio*len(dataset))])
    return trainSet, testSet

# Showcase prediction 
def showPrediction(dataloader, idx, model):
    img, label = next(iter(dataloader))

    # Run image through model
    pred = model(img)
    probs= torch.softmax(pred, dim=1)

    #Gather confidence and prediction
    conf= torch.max(probs, 1)
    pred = torch.argmax(pred[idx], dim=0)

    # Convert to percentage
    acc = conf[idx] * 100

    print(f'Fact: {label[idx]}, Prediction: {(torch.argmax(pred[idx], dim=0))}, Accuracy: {conf[idx] * 100}')
    return img[idx], label[idx], pred, acc 
