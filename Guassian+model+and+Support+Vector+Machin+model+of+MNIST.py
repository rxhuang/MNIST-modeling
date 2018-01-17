
# coding: utf-8

# In[1]:


from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import matplotlib.pylab as plt

# function that imports MNIST data
def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows*cols), dtype=float32)  # Initialize numpy array
    y = zeros((N), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][row] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

 
# function that displays an MNIST image
def displaychar(image):
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()




import numpy as np
from scipy.stats import multivariate_normal
from math import log

# import training data and test data
(xTrain,yTrain) = get_labeled_data("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
(xTest,yTest) = get_labeled_data("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

xValid = xTrain[50000:]
yValid = yTrain[50000:]
x = xTrain[:50000]
y = yTrain[:50000]

c = 3262.5

#compute occurences
pi = []
for i in range(0,10):
    counter = 0
    for j in range(0,50000):
        if(y[j]==i):
            counter += 1
    pi.append(counter)

#compute means, variances
means = zeros((10, 784))
covs = zeros((10,784,784))
for i in range(0,10):
    points = zeros((pi[i], 784))
    counter = 0 
    for j in range(0,50000):
        if(y[j]==i):
            points[counter] = x[j]
            counter += 1
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)+c*np.identity(784)
    means[i] = mean
    covs[i] = cov

# calculate gaussians
gaussian = []
for i in range(10):
    gaussian.append(multivariate_normal(mean=means[i], cov=covs[i]))        
        
# run test points on gaussian and find error rate
wrong = 0
for i in range(len(xTest)):
    pdf = []
    for j in range(10):
        pdf.append(gaussian[j].logpdf(xTest[i]) + log(pi[j], 10))
    if yTest[i] != pdf.index(np.amax(pdf)):
        wrong += 1
print 'error =', long(wrong)/100.0,'%'


# use support vector machine to model MNIST data
from sklearn.svm import SVC
clf = SVC(C=1, kernel='poly', degree=2)
clf.fit(xTrain, yTrain)

train = clf.predict(xTrain)
wrong = 0
for i in range(len(xTrain)):
    if train[i] != yTrain[i]:
        wrong += 1
print float(wrong)/len(xTrain)

train = clf.predict(x)
wrong = 0
for i in range(len(x)):
    if train[i] != y[i]:
        wrong += 1
print float(wrong)/len(x)

sum(clf.n_support_)

