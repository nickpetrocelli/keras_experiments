# Inception-v3 is a very deep DCNN developed by Google, designed to classify 299x299 3 channel images.
# Keras can import this model directly, so we can use it for transfer learning.
# Let's say we have a training set D that has 1,024 input features in 200 output classes.

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the pre-trained InceptionV3 model
# here we do not keep the last four layers, as we want to reshape the input.
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer as first layer
x = Dense(1024, activation='relu')(x)
# an a logistic layer with 200 classes as last layer
predictions = Dense(200, activation='softmax')(x)
# model to train
model = Model(input=base_model.input, output=predictions)

# freeze all convolutional InceptionV3 layers. We don't want to spend the time overwriting their features.
# We only train to modify the features of the input and output dense layers (i.e. to reshape the model to new input).
for layer in base_model.layers: layer.trainable = False

# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train for a few epochs on new data
model.fit_generator(epochs=5, verbose=1)

# freeze some inception blocks
# this is a good hyperparameter to tune
# here we freese first INCEPTION_FREEZE layers and unfreeze the rest
INCEPTION_FREEZE = 172
for layer in model.layers[:INCEPTION_FREEZE]: layer.trainable = False
for layer in model.layers[INCEPTION_FREEZE:]: layer.trainable = True

# Recompile to fine tune optimization

# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# train model again to fine tune the top 2 inception blocks
# alongside the previously added Dense layers
model.fit_generator(epochs=20, verbose=1)
