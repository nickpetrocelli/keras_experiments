# First iteration of a net designed to classify images according to the CIFAR-10 dataset.
# The CIFAR-10 dataset is a set of 60,000 32x32, 3 channel color images split into 10 classes:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

# we will first use a DCNN similar to the LeNet example.


# workaround for a weird OSX bug
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# constants
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = RMSprop()
VALIDATION_SPLIT = 0.2
NB_EPOCH = 20
NB_CLASSES = 10

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

# OHE
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# float and normalize
# I should really make some boilerplate code for this, DRY and all that
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# network
model = Sequential()
# first step: learn 32 3x3 convolutional filters, dropout 25% of values
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# next: dense network with 512 (32 x 32) neurons, then softmax to classify
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# train and test
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

# save to JSON
model_json = model.to_json()
with open('cifar10_architecture.json', 'w') as f:
    f.write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)

# results:
# Test score:  1.0200317998886108
# Test accuracy:  0.6781

# We can do better.
