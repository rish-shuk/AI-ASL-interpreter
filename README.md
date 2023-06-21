# AI-Based Sign Language Recognition App
## <i>A Project by the ISM Team</i>
##### [`Jerome Jose`](https://github.com/jjos425), [`Josef Santos`](https://github.com/JayJsan), [`Rishi Shukla`](https://github.com/rish-shuk)

### About The Project
<i>The following project is an application to recognize static American Sign Language gestures using a selection of user trainable/customizable Convolutional Neural Network (CNN) architectures such as:
* LeNet5
* AlexNet
* Custom built CNN model 

The application also allows webcam input to take images and test trained models on accuracy.</i>

The aim of building this was not only to strengthen our design skills and understanding AI training, but to bring people a stepping stone in which they can use to build, train and test models with simplicity.

<p align="center">
  <img src="https://user-images.githubusercontent.com/71300397/233830184-8558826d-dae4-480a-ac84-e35d7cc63d48.png" />
</p>

The MNIST ASL dataset used can be found on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?datasetId=3258&searchQuery=pytorch).  It is important to note that in this application `J` and `Z` are not included in the training/testing as they are motion-based gestures.

### Tools

##### Acquiring Repo
Application is built upon Python v3.9. 
Download the zip file or clone the repository using: git clone

##### Dependencies
The dependencies to run this application can be found in the `requirements.txt` file.  It is best to create a conda environment that uses Python v3.9, and migrate to the folder with the dependencies to run the command through the terminal: 

`pip install -r requirements.txt`

##### Dependencies
To run program, in terminal run `python3 main.py`

### Future Ventures for Application:
* Allow live ASL detection via webcan to allow motion-based gestures.
* Create a way for users to load their own created models.

### Credits for CSV convert and CNN models:
[VIJAYVIGNESH](https://www.kaggle.com/code/vijaypro/cnn-pytorch-96)

[GEORGY POPOV](https://www.kaggle.com/code/wacholder000/simple-convolution-nn-in-pytorch-test-acc-95)




