import numpy as np
import matplotlib.pyplot as plt
import os

train_dir = "./custom_dataset-1/train/"
val_dir = "./custom_dataset-1/validation/"

train_files = os.listdir(train_dir)
val_files = os.listdir(val_dir)

train_image_paths = []
train_image_labels = []

for i in train_files:
    if i.endswith(".jpg"):
        train_image_paths.append(train_dir + i)
    else:
        train_image_labels.append(train_dir + i)

val_image_paths = []
val_image_labels = []

for i in val_files:
    if i.endswith(".jpg"):
        val_image_paths.append(val_dir + i)
    else:
        val_image_labels.append(val_dir + i)


# GET DATA FROM FILES AND TURN IT INTO ERADABEL DATA


train_images = []
train_labels = []

val_images = []
val_labels = []

IMAGE_SCALE_FACTOR = 4
O_IMAGE_WIDTH = 640
O_IMAGE_HEIGHT = 480

IMG_WIDTH = int(O_IMAGE_WIDTH / IMAGE_SCALE_FACTOR)
IMG_HEIGHT = int(O_IMAGE_HEIGHT / IMAGE_SCALE_FACTOR)

for image in train_image_paths:
    arr = np.array(plt.imread(image))
    arr = arr / 255
    arr = arr[::IMAGE_SCALE_FACTOR, ::IMAGE_SCALE_FACTOR]
    train_images.append(arr)

for image in val_image_paths:
    arr = np.array(plt.imread(image))
    arr = arr / 255
    arr = arr[::IMAGE_SCALE_FACTOR, ::IMAGE_SCALE_FACTOR]
    val_images.append(arr)

for label in train_image_labels:
    arr = np.loadtxt(label)
    train_labels.append(arr[1:])

for label in val_image_labels:
    arr = np.loadtxt(label)
    val_labels.append(arr[1:])

train_images = np.array(train_images)
val_images = np.array(val_images)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

TRAIN = False

if TRAIN:
    print(" ------------- Importing TF ------------- ")

    import tensorflow as tf

    print("------------- Creating Model ------------- ")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='linear')
    ])

    print("------------- Compiling Model ------------- ")

    model.compile(optimizer='adam', loss='mse', metrics='accuracy')

    print("------------- Training Model ------------- ")

    model.fit(train_images, train_labels, epochs=10)

print("------------- Testing Model ------------- ")
from PIL import Image

for i in val_image_paths:
    plt.imshow(train_images[0])

plt.show()

