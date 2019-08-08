from keras import Sequential
from keras.layers import Dense

# big thonk: the book's code is in python 2, I'm coding in python 3 (because it's better, fight me)
# so some things won't look exactly the same.

# sequential model with 1 dense (fully connected graph) layer, 12 neurons, 8 inputs
# initializer: weights are uniformly random small values in (-0.05, 0.05)
# we're not using this for anything, see more complex example below
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))

