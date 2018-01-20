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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/figure/distribution_validation.png)

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

My final model results were: 0.951 
Test set accuracy is 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/downloaded_images/Stop.jpg) 
![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/downloaded_images/Keep_left.jpg) 
![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/downloaded_images/Road_work.jpg) 
![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/downloaded_images/Pedestrians.jpg) 
![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/downloaded_images/Turn_left_ahead.jpg)


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

![alt text](https://github.com/inokatsu/CarND-Traffic-Sign-Recognition/blob/master/figure/softmax_probability.png)


