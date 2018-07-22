# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figure/distribution_validation.png "Visualization"
[image2]: ./downloaded_images/Stop.jpg "Stop sign"
[image3]: ./figure/softmax_probability.png "softmax_probability"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is a link to my [project code](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

I used the numply library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed.


![Visualization][image1]

### Design and Test a Model Architecture

#### 1. Preprocess

As a first step, I decided to convert the images to grayscale because the number of the layer can be reduced and quicken the calculation speed.

As a last step, I normalized the image data so that the data has mean zero and equal variance.


#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 graysclae image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|								 				|
| Max pooling	      	| 1x1 stride, outputs 28x28x6   				|
| Convolution 5x5	    | 1x1 stride, outputs 24x24x16					|
| RELU					|								 				|
| Max pooling	      	| 2x2 stride, outputs 12x12x16  				|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x20					|
| RELU					|								 				|
| Max pooling	      	| 2x2 stride, outputs 5x5x20    				|
| Fully Connected		| Input = 500, Output = 120						|
| Dropout				| 70% keep										|
| Fully Connected		| Input = 120, Output = 84						|
| Relu					|												|
| Dropout				| 70% keep										| 
| Fully Connected		| Input = 84, OUtput = 43						|

#### 3. Train, Validate and Test the Model

##### Parameters
The model was traind with Adam Optimizer.  
Batch size = 128  
Number of epochs = 15  
Learning rate = 0.001  

##### Hyperparameters
mu = 0  
sigma = 0.1  

#### 4. Solution approach

My final model results were: 0.950
Test set accuracy is 0.934

What I did was try and error testing. Following the testing log.

##### Test log and accuracy
1.  init test : __0.893__
2.  change Epoch to 6 : __0.88__
3.  no preprosessing : __0.85__
4.  add one dropout with 10 Epoch: __0.921__
5.  2 dropout : __0.905__
6.  one dropout with rate 0.5: __0.917__
7.  dropout rate 0.6: __0.885__
8.  dropout rate 0.8: __0.924__
9.  Epoch 15: __0.91__
10. learning rate 0.0005: __0.908__
11. Epoch 20 : __0.925__
12. conv 3 layer : __0.952__
13. Epoch 15: __0.950__


 

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

Here are an example of German traffic signs that I found on the web:

![stop sign][image2]

#### 2. Model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop              	| Stop                      					| 
| Keep left    			| Keep left										|
| Road work 			| Road work										|
| Pedestrians   		| Pedestrians			    	 				|
| Turn left ahead    	| Turn left ahead       						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the the Ipython notebook.

![stop sign][image3]
