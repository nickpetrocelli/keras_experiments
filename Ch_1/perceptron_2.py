import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# to reproduce results, you probably knew that already...
np.random.seed(1671)

# building the network
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
# number of possible outputs == number of digits, i.e. |[0-9]|
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
# how much of training set is saved for validation, in this case 20%?
VALIDATION_SPLIT = 0.2

# data is shuffled and split between training and validation sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
# 784 being the number of input features, one per pixel of MNIST image
RESHAPED = 784

# side note: the book doesn't follow PEP 8 at all. Their comment spacing is all over the place. Drives me nuts.
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# final layer is softmax activation function
# 10 output classes, one for each digit
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

# compile.
# using categorical cross-entropy objective function, as that apparently makes the most sense with softmax
# and accuracy as a metric
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# train.
# using NB_EPOCHS epochs (i.e. exposures to training set) and BATCH_SIZE batch size (i.e. number of training cases
# observed before model updates weights)
history = model.fit(X_train,
                    Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

# evaluate on the reserved validation set
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score: ", score[0])
print("Test accuracy ", score[1])
