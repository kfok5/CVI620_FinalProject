# Author:     Kaitlyn Fok
# Student ID: 137237160
# Date:       April 13, 2026
# Project:    Self-Driving Car Simulation Project Using CNN - CVI620 Final Project

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DATA
df = pd.read_csv("driving_log.csv", header=None)

# Column headers for 'steering', 'throttle', 'brake', and 'speed' are missing in csv file
df.columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]

print(df.head())
print(df.columns)

steering = df["steering"]

# MODEL 
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

# Augmenting the data