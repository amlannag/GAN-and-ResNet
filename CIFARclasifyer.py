import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Import data and split testing and training
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Outline the height, width, and number of channels
samples, height, width, channels = X_train.shape

categories = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, categories)
y_test_cat = to_categorical(y_test, categories)

# Normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Calculate mean and std for each channel
mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
std = np.std(X_train, axis=(0, 1, 2), keepdims=True)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    height_shift_range=0.1,
    width_shift_range=0.1,
    fill_mode='reflect'
)

# Fit the data generator to the training data
datagen.fit(X_train)

inputs = Input(shape=(height, width, channels))
net1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(net1)
net2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(net2)
flat = Flatten()(pool2)
net3 = Dense(128, activation="relu")(flat)
output = Dense(categories, activation="softmax")(net3)

model = Model(inputs, output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Using ImageDataGenerator with flow to generate augmented batches
model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
          epochs=10,
          validation_data=(X_test, y_test_cat))