# MNIST-modeling
MNIST is a set of images of hand written digits paired with the actual digits in the images.  
Given an image, my goal is to predict the digit in the image.
First, I used a Guassian model to fit the data. I normalized the data using a constant. The constant is found using cross validation.
After training the data, the model is used on a test set trying to predict digits given images, and the error rate was around 4%.
Then, I used a Soft Vector Machine Model to fit the data, and the error I yield was around 10%.

Files:
t10k-images-idx3-ubyte.gz     test set of 10000 images of hand written digits
t10k-labels-idx1-ubyte.gz     test set of 10000 digits corresponding to the images in the test set
train-images-idx3-ubyte.gz    training set of 60000 images of hand written digits
train-labels-idx1-ubyte.gz    training set of 60000 digits corresponding to the images in the training set
