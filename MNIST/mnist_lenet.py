import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
 

np.random.seed(123)  # for reproducibility
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
 
# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print("Training matrix shape", X_train.shape)
print("Training target matrix shape", Y_train.shape)
print("Testing matrix shape", X_test.shape)
print("Testing target matrix shape", Y_test.shape)


print 'CC4' 
model = Sequential()
model.add(Convolution2D(20, 5, strides=(1, 1), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), data_format='channels_first'))
model.add(Convolution2D(50, 5, strides=(1, 1), activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1), data_format='channels_first'))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
 
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
 
print 'A'
model.fit(X_train, Y_train, batch_size=8, epochs=20, verbose=2)
 
score = model.evaluate(X_test, Y_test, verbose=0)
print score[1]

