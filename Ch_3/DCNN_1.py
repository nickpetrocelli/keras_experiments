# Chapter 3 focuses on Deep Convolutional Neural Networks (DCNNs).
# The key difference between this and a basic dense net (as in chapter 1) is that rather than breaking down
# the input into its smallest quantums (e.g. pixels in an image) and using each as an input feature,
# each input feature is instead a subsample of the overall input, which itself is subsampled in deeper layers.

# To intuit this, picture an image as a matrix, and the convolution process as sliding a small picture frame across
# it to create submatrices.
# The size of a submatrix is called the stride length in Keras.

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D


# sequential model with a convolutional layer
# output dimensionality of 32
# extension of each filter is 3x3
# input image has shame (256, 256) on 3 channels
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(256, 256, 3)))

# The set of subsamples generated by convolution of an input is called a feature map.
# To summarize the output of a feature map
# (i.e. create a cohesive meaning from observations on a subsample of the input) we use a pooling layer.

# Here we add a "max-pooling" layer, a simple layer that takes the maximum value(s) from a map.
# Here its size is (2,2), meaning tha the map will be coalesced into a matrix of maximum values from different
# regions of the feature map, with half the length and width of the pooled map.

model.add(MaxPooling2D(pool_size=(2, 2)))
