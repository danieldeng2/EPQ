import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 66, 200, 3])

    # Convolutional Layer #1 & Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=24,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 & Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=36,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Convolutional Layer #3 & Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=48,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.elu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.elu)
    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.elu)
    dropout = tf.layers.dropout(
        inputs=conv5, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout_flat = tf.reshape(dropout, [-1, 1216])
    dense1 = tf.layers.dense(inputs=dropout_flat, units=100, activation=tf.nn.elu)
    dense2 = tf.layers.dense(inputs=dense1, units=50, activation=tf.nn.elu)
    dense3 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.elu)
    dense4 = tf.layers.dense(inputs=dense3, units=1)
    result = tf.reshape(dense4, [-1], name="result")
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=result)

    loss = tf.losses.mean_squared_error(labels=labels,predictions=result)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=result)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    for track_n in range(1, 3):
        for try_n in range(1, 6):
            #Read the images and load the data.
            file_dir = "/home/daniel/Projects/sdving/track" + str(track_n) + "/try" + str(try_n) + "/driving_log.csv"
            print("file_dir: " + file_dir)
            driving_sample = pd.read_csv(file_dir, sep="," , names = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed'])
            print(driving_sample.describe())
            image_data = []
            for index, row in driving_sample.iterrows():
                image_data.append(cv2.resize(cv2.cvtColor(cv2.imread(row["Center"])[60:-25, :, :], cv2.COLOR_RGB2YUV),(200, 66),cv2.INTER_AREA))
                print("loaded: " + row["Center"])
            print('image_data shape:', np.array(image_data).shape)

            train_images, test_images, train_labels, test_labels = train_test_split(np.array(image_data),np.array(driving_sample['Steering'].values, dtype=np.float32), test_size=0.2, random_state=0)

            for i in range(1,len(train_images)):
                train_images[i] = train_images[i]/127.5-1.0
            for i in range(1,len(test_images)):
                test_images[i] = test_images[i]/127.5-1.0
            # plt.imshow(train_images[0])
            # plt.show()
            print('train_images shape: ', train_images.shape)
            print('train_labels shape: ', train_labels.shape)
            # Create the Estimator
            behaviour_regressor = tf.estimator.Estimator(
                model_fn=cnn_model_fn, model_dir="/home/daniel/Projects/EPQ/Behaviour-cloning-model")

            logging_hook = tf.train.LoggingTensorHook(
                     {"predictions" : "result"}, every_n_iter=50)
            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_images},
                y=train_labels,
                batch_size=1,
                num_epochs=50,
                shuffle=True)
            behaviour_regressor.train(
                input_fn=train_input_fn,
                steps=15000
                 ,hooks=[logging_hook]
                )

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": test_images},
                y=test_labels,
                batch_size=1,
                num_epochs=1,
                shuffle=False)
            eval_results = behaviour_regressor.evaluate(input_fn=eval_input_fn)
            print(eval_results)
            print("trained with track ", track_n, "try ", try_n)
