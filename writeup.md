# Traffic Sign Recognition 

In this project I trained a deep neural network to classify images of german traffic signs as an assignment of the udacity self driving car nanodegree. I augmented the images with random brightness, rotations, scaling, and shearing. I used the LeNet-5 architecture with more filters and dropout. I've got an accuracy on the test set of 95,7 %.

---

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/FabianHertwig/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 observations.
* The size of the validation set is 12630 observations.
* The size of test set is 4410 observations.
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are multiple bar charts showing how the classes are distrubuted for the train, validation and test set.

Class distribution is pretty much the same on each set, though there are more prominent classes than others. To improve results further, one could rebalance the class distribution with augmented images.

![class distributions](./writeup_sources/class_distributions.png)

I also plotted some images for each class of the dataset to get an overview of what the images show. This is only a part of the plot, please view the notebook to see all classes.

![class examples](./writeup_sources/class_examples.png)

The class distribution seem to be according to the distribution of the signs you see on german streets. For example class one, a speed limit of 20 km/h is very rare to see, as usally the speed limit in residential areas is 30 km/h. So there are only 180 images of this sign. There are up to 2000 images for the "popular" signs.

As far as I know the images were taken from a car and multiple images of each sign were taken as the car moved by. Therfore the dateset contains the same image with different alterations in the angle or in brightness. Lets take a look at random pictures per class to see some different ones. Again, this is only a part of the plot, please view the notebook to see all classes.

![class examples shuffled](./writeup_sources/class_examples_shuffled.png)

### Design and Test a Model Architecture

#### 1. Preprocessing and Augmentation


To train the net I prepocessed the images and also added augmented images.

For preprocessing the images are normalized, so every image uses the full brightness spectrum from value 0 to value 255. This improves images that are very dark, but also adds shifts in colour. One could use better techniques to normalize the brightness, as converting in the HSL colorspace and normalize brighntess there. 

Normalizing the images improved the accuracy of the net.

For augmentation random brightness got added and then the images got normalized. Brightness is added before normalizing, to emulate different conditions when the picture was taken. Also random translations, rotations, sheering and scaling were added. These augmentation techniques are used quite often with image recognition tasks and should improve overfitting. For example the  in the famous [ImageNet Classification with Deep Convolutional Neural Networks paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) image translations, mirroring and changes in colour were used. I decided not to mirror the images as you would not encounter a mirrored sign in the real world.

At last the image values where converted to a -1.0 to 1.0 range for deep learning.

Following are examples for the augmentations steps.

Original Image:  
![original image](./writeup_sources/augmentation/1orig.png)

Random Brightness added to the Image:  
![Random Brightness](./writeup_sources/augmentation/2bright.png)

Pixel intensity normalized:  
![normalized image](./writeup_sources/augmentation/3normalize.png)

Shear, translation, rotation and scale added to the Image. This is done in one step using a transformation matrix.  
![warped image](./writeup_sources/augmentation/3warp.png)


For each image in the original training set I added five augmented images.

Here are a few more examples:
![warped images](./writeup_sources/augmentation/examples.png)


#### 2. The final model architecture

Below is a table of the architecture. It is the same as the LeNet-5 architecture with more filters and therefore more fully connected nodes.

| Layer           | Description                                 |
|-----------------|---------------------------------------------|
| Input           | 32x32x3 Image Data                          |
| Convolution 5x5 | Valid Padding, 1x1 Strides, 28x28x16 Output |
| Relu            |                                             |
| Max Pool        | 2x2 Strides, 14x14x16 Output                |
| Convolution 5x5 | Valid Padding, 1x1 Strides, 10x10x32 Output |
| Relu            |                                             |
| Max Pool        | 2x2 Strides, 5x5x32 Output                  |
| Flatten         | 800 Output                                  |
| Fully Connected | 200 Output                                  |
| Relu            |                                             |
| Fully Connected | 100 Output                                  |
| Relu            |                                             |
| Dropout         | 50% Keep Probability                        |
| Fully Connected | Output 43                                   |

#### 3. Hyperparameters and training approach.
I used the same architecture as the LeNet-5 net, as it was build for image classification tasks. Also most of the code got implemented earlier in the nanodegree so it seemed obvious to reuse the code.

To improve the accuracy of the net, I increased the filters and therefore I had to increase the number of fully connected neurons accordingly. At first I started with quite a lot of filters, and a lot of neurons. Then I decreased the filters step by step to prevent overfitting  until I got good results. 

I tried to create an overfitting model first, so I know that the model is able to learn the concepts. Then I tried to decrease overfitting by decreasing the number of filters, adding a dropout layer with a 50% keep probability and image augmentation.

I used the **adam optimizer** with its default **learning rate of 0.001**. I tried to lower the training rate, but that seemed to increase training time without accuracy improvements. The adam optimizer should addapt the training rate anyways.

I also tried to increase the batch size until I run in memory issues, but that seemed to lower the accuracy somehow. So I kept the **batch size at 512**. The model ran for **30 Epochs**, but it didn't improve after 20.

As a final result I got a validation accuracy of 0.951 and test accuracy of 0.957. 

In the loss chart below you can see that validations loss starts to rise again after about 10 Epochs of training. That shows that the net is still overfitting, but I'm happy with the results it produces anyways.

![warped images](./writeup_sources/train_loss_graph.png)


This are examples for the most confused classes from most confusions to less confusions. The signs are very similiar indeed. 

![warped images](./writeup_sources/top_confused1.png)  
![warped images](./writeup_sources/top_confused2.png)  
![warped images](./writeup_sources/top_confused3.png)  
![warped images](./writeup_sources/top_confused4.png)



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

As I life in Germany, I drove around and took some images. I mounted my Phone behind my windshield to take the images. The windscreen was quite dirty, from time to time it was raining. So the images will be challenging. Also there is one 120 km/h speed limit image which was displayed with LEDs and the signs background is black instead of white. Im curious if that one gets recognized by the net.

Here is one example how the image was taken and for the dirty windshield in the corner :)

![camera position](./images/example.jpg)

Here are the images I have chosen:

![my signs](./writeup_sources/own_images_examples.png)


#### 2. Predictions power on new traffic signs

Here are the results of the prediction:

|**Prediction**|**True Class**|**SignName Prediction**|**SignName True Class**
:-----:|:-----:|:-----:|:-----:|:-----:
|12|12|Priority road|Priority road
|18|18|General caution|General caution
|18|18|General caution|General caution
|2|2|Speed limit (50km/h)|Speed limit (50km/h)
|3|3|Speed limit (60km/h)|Speed limit (60km/h)
|5|8|Speed limit (80km/h)|Speed limit (120km/h)
|8|8|Speed limit (120km/h)|Speed limit (120km/h)

![sign and prediction](./writeup_sources/own_image_prediction1.png)  
![sign and prediction](./writeup_sources/own_image_prediction2.png)  
![sign and prediction](./writeup_sources/own_image_prediction3.png)  
![sign and prediction](./writeup_sources/own_image_prediction4.png)  
![sign and prediction](./writeup_sources/own_image_prediction7.png)  
![sign and prediction](./writeup_sources/own_image_prediction5.png)  
![sign and prediction](./writeup_sources/own_image_prediction6.png)  




The model confused some of the images. I really don't know how it confused class "General caution" with "No entry". To confuse the very different Speed Limit (120km/h) sign seems reasonable. 

On these images the model got an accuracy of 71% which is worse than on the testset. But these images are harder to classify bacause of the noise added by the rain and there is one sign that the model has never seen before.

#### 3. Softmax Probabilities

Below you can see the top 5 class propabilities for each of my own images.
The model has lower probabilities on "general caution" sign, as there are very similiar signs with also higher probabilities. Also the led speed limit sign has low probabilities as the model has never seen such a led sign.

![sign and prediction](./writeup_sources/own_image_prediction1.png)

![sign and prediction](./writeup_sources/softmaxprobs1.png)

---

![sign and prediction](./writeup_sources/own_image_prediction2.png)

![sign and prediction](./writeup_sources/softmaxprobs2.png)

---

![sign and prediction](./writeup_sources/own_image_prediction3.png)

![sign and prediction](./writeup_sources/softmaxprobs3.png)

---


![sign and prediction](./writeup_sources/own_image_prediction4.png)

![sign and prediction](./writeup_sources/softmaxprobs4.png)

---


![sign and prediction](./writeup_sources/own_image_prediction7.png)

![sign and prediction](./writeup_sources/softmaxprobs5.png)

---

![sign and prediction](./writeup_sources/own_image_prediction5.png)

![sign and prediction](./writeup_sources/softmaxprobs6.png)

---


![sign and prediction](./writeup_sources/own_image_prediction6.png)

![sign and prediction](./writeup_sources/softmaxprobs7.png)

---


### Visualizing the Neural Network 

Below you can see the activation map for the relu output of the first convolutional layer.

In this activations, you can clearly recognize the number, the round shape and that there is a ring. FeatureMap 8 seems to look for a dark sign in front of a bright background.

![feature map](./writeup_sources/featuremap1.png)

With this other sign but still activations from layer one it is really hard to recognize the numbers and the shape of the sign. Though the model still predicted the correct class.

![feature map](./writeup_sources/featuremap2.png)

This is the 120km/h sign displayed with leds, that got confused with another image. I wonder why, as the activation map doesn't seem too different to the other activation maps of speed limit signs.

![feature map](./writeup_sources/featuremap3.png)

This is the seconds layer activiation. You can't really see a lot, as the resolution is very low. Maybe I should try to remove the maxpool layers as they lower the resolution... another time :)

![feature map](./writeup_sources/featuremap4.png)





