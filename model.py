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
    train_images[i] = train_images[i]/127.5-1.0
for i in range(1,len(test_images)):
    test_images[i] = test_images[i]/127.5-1.0
# plt.imshow(train_images[0])
# plt.show()
print('train_images shape: ', train_images.shape)
print('train_labels shape: ', train_labels.shape)

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/home/daniel/Documents/EPQ/Behaviour-cloning-model")

#TODO: set up logging hook, change training & eval parameters
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_images},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_images},
    y=test_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)


def cnn_model_fn(images, labels, mode):
    input_layer = tf.reshape(images["x"], [-1, 75, 320, 3])

    # Convolutional Layer #1 & Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=24,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 & Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=36,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3 & Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=48,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.elu)

    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.elu)

    dropout = tf.layers.dropout(
        inputs=conv5, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dropout_flat = tf.reshape(dropout, [-1, 1152])
    dense1 = tf.layers.dense(inputs=dropout_flat, units=100, activation=tf.nn.elu)
    dense2 = tf.layers.dense(inputs=dense1, units=50, activation=tf.nn.elu)
    dense3 = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.elu)
    result = tf.layers.dense(inputs=dense3, units=1, activation=tf.nn.elu)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=result)

    #TODO: change the loss function
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=result)

    #TODO: change optimizer and learning rate
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=result)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
