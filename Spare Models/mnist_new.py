from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K

batch_size = 128
nb_classes = 10
nb_epoch = 6

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


# normalize data
X_train /= 255
X_test /= 255

# handling compatibility issues with image shape in theano and tensoflow backend
if K.image_data_format() == "th":
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# create model
model = Sequential()
model.add(
    Convolution2D(nb_filters, kernel_size[0], kernel_size[1], input_shape=input_shape,)
)
model.add(Activation("relu"))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))  # add dropout for better results

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))  # add dropout for better results
model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
)

model.fit(
    X_train, Y_train, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test),
)
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test cost:", score[0])
print("Test accuracy:", score[1])

print("[INFO] saving model to disk...")

# save model to disk
model.save("output/mnist_cnn.model")
