
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense, GlobalAveragePooling2D
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

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        # If stride is not 1 or in_planes != planes, apply a shortcut
        if stride != 1 or in_planes != planes:
            self.shortcut = tf.keras.Sequential([
                Conv2D(planes, kernel_size=1, strides=stride, use_bias=False),
                BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x  # Identity shortcut

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out += self.shortcut(x)
        out = ReLU()(out)
        
        return out

class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        
        # Build layers using _make_layer
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final classification layer
        self.global_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        out = self.bn1(self.conv1(x), training=training)
        out = ReLU()(out)
        
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        out = self.global_pool(out)
        out = self.fc(out)
        return out

# Instantiate the ResNet18 model (BasicBlock and [2, 2, 2, 2] layers)
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

model = ResNet18()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


def piecewise_lr_schedule(epoch, lr):
    if epoch < 15:
        return 0.005 + (0.1 - 0.005) * (epoch / 15)  # Increase learning rate from 0.005 to 0.1
    elif epoch < 30:
        return 0.1
    else:
        return 0.1 * (1 - (epoch - 30) / 15)  # Decrease learning rate linearly after epoch 30

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(piecewise_lr_schedule)


model.fit(datagen.flow(X_train, y_train_cat, batch_size=128),
          epochs=50,
          validation_data=(X_test, y_test_cat),
          callbacks=[lr_scheduler])

predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
print(predictions)


print(classification_report(y_test, predictions))