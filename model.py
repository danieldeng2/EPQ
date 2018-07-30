import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from PIL import Image
import cv2
#from scipy.misc import imread
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


#Read the images and load the data.
driving_sample = pd.read_csv("~/Documents/EPQ/track1_1/driving_log.csv", sep="," , names = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed'])
print(driving_sample.describe())
image_data = []
for index, row in driving_sample.iterrows():
    image_data.append(cv2.cvtColor(cv2.imread (row["Center"])[60:-25, :, :], cv2.COLOR_RGB2YUV))
    print("loaded: " + row["Center"])
print('image_data shape:', np.array(image_data).shape)

train_images, test_images, train_labels, test_labels = train_test_split(np.array(image_data), driving_sample['Steering'].values, test_size=0.2, random_state=0)
for i in range(1,len(train_images)):
    train_images[i] = train_images[i]/255.0
# plt.imshow(train_images[0])
# plt.show()
print('train_images shape: ', train_images.shape)
print('train_labels shape: ', train_labels.shape)
