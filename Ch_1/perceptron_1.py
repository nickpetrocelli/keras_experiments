from keras import Sequential
from keras.layers import Dense

# big thonk: the book's code is in python 2, I'm coding in python 3 (because it's better, fight me)
# so some things won't look exactly the same.
# also I try to be funny in my comments. This is really for my benefit but I'll keep it PG, don't worry.

# sequential model with 1 dense (fully connected graph) layer, 12 neurons, 8 inputs
# initializer: weights are uniformly random small values in (-0.05, 0.05)
# we're not using this for anything, see more complex example in perceptron_2
# (also I like the royal 'we' in my comments)

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))

# oh, and by the way:
# I'll be using the TensorFlow backend here, there, and everywhere. Unless otherwise noted.
