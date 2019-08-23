# Chapter 4 explores the concept of a Generative Adversarial Network, or GAN.
# GANs provide a means to "forge" data that looks nearly identical to human-created data,
# such as images or sounds.

# GANs work by simultaneously training two different neural networks: a Generator G, and a descriminator (or decider) D.
# G is a network that generates forged data based on random input, and D is a network that accepts said data
# and tries to determine whether it is a forgery (along with 'real' data for comparison).

# As both nets are backpropagated using a gradient descent function,
# improvements in G cause improvements in D, and vice versa.

# The nets implemented here form a Deep Convolutional Generative Adversarial Network or DCGAN.
# As the name implies, the generator here uses convolutional layers to transform the random input into a full output,
# in this case a 64x64 image. This does not require any dense or pooling layers to be used (aside from the input layer).

# the book outlines this in the Keras 1 syntax (for some reason) but I'm going to try to convert it to Keras 2
# as an exercise for myself, and to maintain my own sanity.


from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2D, MaxPooling2D, Flatten


# define the generator
def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization)
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


# define the discriminator
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


