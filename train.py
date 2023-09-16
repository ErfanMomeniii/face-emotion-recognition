from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Input
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, RandomFlip, RandomRotation, AveragePooling2D
import cv2
import glob
import numpy as np

labels = [
    'angry',
    'disgusted',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised'
]

model = Sequential


def train():
    global labels, model

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for a, i in enumerate(labels):
        for train_image in glob.iglob("./dataset/train/%s/*" % i):
            im = cv2.imread(train_image)
            im = im.astype(np.float64) / 255
            train_images.append(np.array(im))
            train_labels.append(a)

    add_to_train = 1
    for a, i in enumerate(labels):
        for test_image in glob.iglob("/dataset/test/%s/*" % i):
            if add_to_train % 10 != 0:
                im = cv2.imread(test_image)
                im = im.astype(np.float64) / 255
                train_images.append(np.array(im))
                train_labels.append(a)
            else:
                im = cv2.imread(test_image)
                im = im.astype(np.float64) / 255
                test_images.append(np.array(im))
                test_labels.append(a)
            add_to_train += 1

    model = Sequential([
        Input(shape=(48, 48, 3)),
        Conv2D(64, kernel_size=(3, 3), use_bias=False, activation="relu"),
        MaxPooling2D(pool_size=3),
        Flatten(),
        Dense(len(labels), activation='softmax'),
    ])

    model.compile(SGD(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        np.array(train_images),
        np.array(train_labels),
        batch_size=1,
        epochs=200,
        validation_data=(np.array(test_images), np.array(test_labels)),
    )
