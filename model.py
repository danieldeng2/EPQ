import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

#Read the images and load the data.
driving_sample = pd.read_csv("~/Documents/EPQ/track1_1/driving_log.csv", sep="," , names = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed'])
print(driving_sample.describe())
Center_Images = pd.Series([])
image_data = []
i = 0
for index, row in driving_sample.iterrows():
    i+=1
    if i == 5:
        break
    image_data.append(Image.open(row["Center"]))
    print("loaded: " + row["Center"])
Center_Images = pd.Series(image_data)
