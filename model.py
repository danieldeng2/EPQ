import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image

driving_sample = pd.read_csv("~/Documents/EPQ/Mine/track1_1/driving_log.csv", sep="," , names = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed'])
print(driving_sample.describe())
Center_Images = pd.Series([])
for index, row in driving_sample.iterrows():
    Center_Images = Image.open(row["Center"])
    print("loaded: " + row["Center"])
