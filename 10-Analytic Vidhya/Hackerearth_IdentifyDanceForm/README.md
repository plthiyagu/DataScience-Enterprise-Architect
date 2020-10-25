# Identify the INDIAN classical dance form
Classifying 8 different dance forms using transfer learning, while utilizing pre-trained model ResNet50.


## Problem statement
```
This International Dance Day, an event management company organized an evening of Indian classical dance 
performances to celebrate the rich, eloquent, and elegant art of dance. 
The task is to build a deep learning model that can help the company classify these images into 
eight categories of Indian classical dance.
```
The eight categories of Indian classical dance are as follows:
* **Manipuri**
* **Bharatanatyam**
* **Odissi**
* **Kathakali**
* **Kathak**
* **Sattriya**
* **Kuchipudi**
* **Mohiniyattam**




<br />

## Data Description
This data set consists of the following two columns:

Column Name | Description
------------- | -------------
Image  | Name of Image
Target  | 	Category of Image ['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam']

The data folder consists of two folders and two .csv files. The details are as follows:
* **train** : Contains 364 images for 8 classes
* **test**: Contains 156 images
* **train.csv**: 364 x 2
* **test.csv**: 156 x 1




## Architecture 
Given below shows the basic architecture implemented along with ResNet50.
<img src="https://github.com/AshishGusain17/Hackerearth-Deep-Learning-Challenge/blob/master/model_plot.png?raw=true" width="410">



## Working
* The input images used are all 224 x 224 x 3 in shape.
* The end layers of the ResNet50 are removed.
* The last convolutional layer left is completely flattened into a functional layer.
* The end 10 layers are unfreezed and their weights changes while training.
* Now, 2 more functional layers of 1024 and 512 neurons are added.
* Both of them includes a dropout of 50% neurons in each propagation.
* At the end, we have a softmax classification layer with 8 neurons to predict output.



## Results
```
The test accuracy finally received is 82.5%.
As the number of training images are only 356 and the number of classes are 8, 82.5% seems to be 
a decent accuracy. Image augmentation played a major role in the entire training.
Exact code can be seen in danceForms.ipynb file.
```
