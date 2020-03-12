from __future__ import division, print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
import os

BATCH_SIZE = 128
NUM_EPOCHS = 20
MODEL_DIR = "C:/Users/Owner/Desktop/AI勉強/直観DeepLearning/ch02/tmp"

(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
Xtrain = Xtrain.reshape(60000, 784).astype("float32") / 255
Xtest = Xtest.reshape(10000, 784).astype("float32") / 255
Ytrain = np_utils.to_categorical(ytrain, 10)
Ytest = np_utils.to_categorical(ytest, 10)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784, ), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))

logs = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                            batch_size=32, write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None)

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
         validation_split=0.1, callbacks=[checkpoint, logs])

keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                            batch_size=32, write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None)

