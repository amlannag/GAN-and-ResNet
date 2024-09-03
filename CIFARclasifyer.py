#Importing libraries 
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

#Import data and split testing and training
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Outline the height, width and number of channels 
samples, height, width, channels = X_train.shape

categories = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, categories)
y_test_cat = to_categorical(y_test, categories)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)

mean = tf.reduce_mean(X_train_tensor, axis=(0, 1, 2))
std = tf.math.reduce_std(X_train_tensor, axis=(0, 1, 2))

# Convert to numpy arrays for easy viewing
mean_np = mean.numpy()
std_np = std.numpy()

print("Mean of each channel (R, G, B):", mean_np)
print("Standard deviation of each channel (R, G, B):", std_np)

X_train_tensor = (X_train_tensor - mean_np) / std_np
X_test_tensor = (X_test_tensor - mean_np) / std_np

datagen = ImageDataGenerator(
    horizontal_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    fill_mode='reflect'
)

# Fit the data generator to the training data
datagen.fit(X_train_tensor)

inputs = Input(shape=(height, width, channels))
net1 = Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation="relu")(inputs)
pool1 = MaxPooling2D(pool_size=(2,2))(net1)
net2 = Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2,2))(net2)
flat = Flatten()(pool2)
net3 = Dense(128, activation="relu")(flat)
output = Dense(categories, activation="softmax")(net3)

model = Model(inputs, output)

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_tensor, y_train_cat,
                epochs=2,
                batch_size=4,
                shuffle=True,
                validation_data=(X_test_tensor, y_test_cat))