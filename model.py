import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
CSV_COLUMN_NAMES = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Break', 'Speed']

def cnn_model_fn(features, labels, mode):



def load_image(path):
    image = Image.open(path.strip())
    # Normalize the image pixels to range -1 to 1.
    image = np.array(image, np.float32)
    image /= 127.5
    image -= 1.
    # Slice off the top and bottom pixels to remove the sky
    image = image[40:130, :]
    return image    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',type=str)
    args = parser.parse_args()
    print('Training with ' + args.data_dir);
    data = pd.read_csv(args.data_dir, names=cols, header=1)
    images = data[['Center', 'Left', 'Right']]
    angles = data['Steering']

    #left_images = []
    center_images = []
    #right_images = []
    batch_angles = []
    for i in np.arange(len(images)):
        center_image = load_image(images.iloc[i]['Center'])
        #left_image = load_image(images.iloc[i]['Left'])
        #right_image = load_image(images.iloc[i]['Right'])

        center_images.append(center_image)
        #left_images.append(left_image)
        #right_images.append(right_image)
        batch_angles.append(float(angles.iloc[i]))

    images_train, images_evaluation, angles_train, angles_evaluation = train_test_split(center_images, batch_angles, test_size=0.2, random_state=0)
    steering_predictor = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/driving_convnet_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images_train},
        y=batch_angles,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    steering_predictor.train(
        input_fn=train_input_fn,
        steps=len(images))

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images_evaluation},
        y=angles_evaluation,
        num_epochs=1,
        shuffle=False)
    eval_results = steering_predictor.evaluate(input_fn=eval_input_fn)
    print(eval_results)







    
    
















if __name__ == '__main__':
    main()