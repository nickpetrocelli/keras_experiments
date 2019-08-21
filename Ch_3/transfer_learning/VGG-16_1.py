# VGG-16 is a 16 layer DCNN created by Simonyan et al.
# It is used to classify images in the ImageNet ILSVRC-2012 dataset, a dataset consisting of
# 1,000 classes of 224x224 images on 3 channels.

# Here I will extract features from the VGG-16 model to 'pre-train' a new image recognition DCNN.
# This is a practice called 'Transfer Learning'.

# pre-defined model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np

# pre-built and pre-trained deep learning VGG16 model
base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.output_shape)

# extract features from block4_pool block
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# get the features from this block
features = model.predict(x)
