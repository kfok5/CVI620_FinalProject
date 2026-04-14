# Author:     Kaitlyn Fok
# Student ID: 137237160
# Date:       April 13, 2026
# Project:    Self-Driving Car Simulation Project Using CNN - CVI620 Final Project

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
from sklearn.model_selection import train_test_split
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

# DATA
df = pd.read_csv("driving_log.csv", header=None)

# Column headers for 'steering', 'throttle', 'brake', and 'speed' are missing in csv file
df.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

print(df.head())
print(df.columns)

steering = df["steering"]

turning = df[abs(df["steering"]) > 0.05]
straight = df[abs(df["steering"]) <= 0.05]

straight = straight.sample(n=len(turning))

balanced_df = pd.concat([turning, straight])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df["steering"], bins=25)
plt.title("Before balancing")

plt.subplot(1,2,2)
plt.hist(balanced_df["steering"], bins=25)
plt.title("After balancing")

plt.show()

X = balanced_df["center"]
y = balanced_df["steering"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# Data augmentation
def load(img_path):
    filename = os.path.basename(img_path)
    path = os.path.join("IMG", filename)
    return cv2.imread(path)

def flip(img, steering):
    img = cv2.flip(img, 1)
    return img, -steering

def adjust_bright(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] = hsv[:,:,2] * r
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def zoom(img):
    h, w = img.shape[:2]
    zm = 1 + 0.2 * np.random.rand()
    hz, wz = int(h/zm), int(w/zm)
    y1 = np.random.randint(0, h - hz)
    x1 = np.random.randint(0, w - wz)
    cropped = img[y1:y1+hz, x1:x1+wz]
    return cv2.resize(cropped, (w, h))

def pan(img):
    h, w = img.shape[:2]
    x_shift = int(0.1 * w * (np.random.rand() - 0.5))
    y_shift = int(0.1 * h * (np.random.rand() - 0.5))

    mtx = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(img, mtx, (w, h))

def rotate(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-10, 10)

    mtx = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, mtx, (w, h))

def augmenting(image, steering):
    if np.random.rand() < 0.5:
        image, steering = flip(image, steering)

    if np.random.rand() < 0.5:
        image = adjust_bright(image)

    if np.random.rand() < 0.5:
        image = zoom(image)

    if np.random.rand() < 0.5:
        image = pan(image)

    if np.random.rand() < 0.5:
        image = rotate(image)

    return image, steering


# Data pre-processing
def preprocessing(image):
    h, w, _ = image.shape
    image = image[int(h * 0.4):int(h * 0.85), :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255

    return image


# Batching dataset
def data_generator(img_path, steer_angle, batch_size=32, training=True):
    while True:
        for i in range(0, len(img_path), batch_size):

            batch_img = []
            batch_steer = []

            for x in range(i, min(i + batch_size, len(img_path))):
                img = load(img_path.iloc[x])
                angle = steer_angle.iloc[x]

                if training:
                    img, angle = augmenting(img, angle)

                img = preprocessing(img)
                batch_img.append(img)
                batch_steer.append(angle)

            yield np.array(batch_img), np.array(batch_steer)

# MODEL
def self_driving_model():

    model = Sequential()

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu', input_shape=(66,200,3)))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

model = self_driving_model()

train_batch = data_generator(X_train, y_train, training=True)
validate_batch  = data_generator(X_val, y_val, training=False)

H = model.fit(
    train_batch,
    steps_per_epoch = len(X_train)//32,
    validation_data = validate_batch,
    validation_steps = len(X_val)//32,
    epochs=3
)

# EVALUATE
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

model.save("self_driving_model.h5")