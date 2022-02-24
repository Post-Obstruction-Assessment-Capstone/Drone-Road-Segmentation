# Author: Trevor Sherrard
# Since: Feb. 21, 2022
# Purpose: This file contains functionallity needed to run inference on a single image

import cv2
import numpy as np
import tensorflow as tf
import keras

# declare file paths
model_file_loc = "../models/saved_unet_model.h5"
test_image_loc = "../dataset/semantic_drone_dataset/original_images/000.jpg"

# declare goal image sizes
img_height = 800
img_width = 1200

def preprocess_image(img_file):
    def func(img_file):
        img_file = img_file.decode()
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    image = tf.convert_to_tensor(tf.numpy_function(func, [img_file], [tf.float32]))
    image = tf.reshape(image, (img_height, img_width, 3))
    return image

def load_image_as_dataset(img_file):
    dataset = tf.data.Dataset.from_tensor_slices(img_file)
    dataset = dataset.map(preprocess_image)
    dataset = dataset.batch(1)
    return dataset

def run_inference(image_loc):
    # load image
    dataset = load_image_as_dataset([image_loc])

    # load model
    model = keras.models.load_model(model_file_loc)

    # run inference
    pred = model.predict(dataset)
    predictions = np.argmax(pred, axis=3)
    cv2.imshow("test", predictions)
    cv2.waitKey(0)

if(__name__ == "__main__"):
    run_inference(test_image_loc)
