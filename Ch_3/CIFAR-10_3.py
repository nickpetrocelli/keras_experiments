# Third iteration of a net designed to classify images according to the CIFAR-10 dataset.
# This is improved by augmenting the dataset with transformed images (rotation, rescaling, flip, etc.)

# The CIFAR-10 dataset is a set of 60,000 32x32, 3 channel color images split into 10 classes:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.



# workaround for a weird OSX bug
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import tempfile
import shutil
import matplotlib.pyplot as plt

# augmentation param
from keras.preprocessing.image import ImageDataGenerator
NUM_TO_AUGMENT = 5


IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# constants
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = RMSprop()
VALIDATION_SPLIT = 0.2
NB_EPOCH = 50
NB_CLASSES = 10

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

# Augmenting
print("I never asked for this...")
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

xtas, ytas = [], []
tmpdir = tempfile.mkdtemp()
print('Temp directory: ', tmpdir)
for i in range(X_train.shape[0]):
    num_aug = 0
    x = X_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in datagen.flow(x, batch_size=1, save_to_dir=tmpdir, save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug+=1

datagen.fit(X_train)

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
# conv -> conv -> maxpool -> dropout
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv (64 filters) -> conv -> maxpool -> dropout
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
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
dir = 'cifar10_3_model'
with open(os.path.join(dir, 'cifar10_architecture.json'), 'w') as f:
    f.write(model_json)
model.save_weights(os.path.join(dir, 'cifar10_weights.h5'), overwrite=True)

# cleanup
shutil.rmtree(tmpdir)
