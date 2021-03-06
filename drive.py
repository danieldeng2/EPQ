import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from model import cnn_model_fn
sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 1
MIN_SPEED = 0.01

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data["speed"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"]))).convert('RGB')

        try:
            image = np.asarray(image)
            image = image[:, :, ::-1].copy()
            image = cv2.resize(cv2.cvtColor(image[60:-25, :, :], cv2.COLOR_RGB2YUV),(200, 66),cv2.INTER_AREA)
            image = np.array([image])
            # print(image.shape)
            # plt.imshow(image[0])
            # plt.show()

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": image},
                batch_size=1,
                shuffle=False)
            result = behaviour_regressor.predict(input_fn=predict_input_fn)
            steering_angle = next(result)[0]

            global speed_limit
            speed_limit = MIN_SPEED if speed > speed_limit else MAX_SPEED

            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, 0.1)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()}, skip_sid=True)


if __name__ == '__main__':

    behaviour_regressor = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./Behaviour-cloning-model")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
