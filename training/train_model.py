import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from unet_model import multi_unet_model
import pandas as pd

# set various parameters required for training
num_classes = 23
img_height = 800
img_width = 1200
num_epochs = 12
batch_size = 5
prefetch = 2
min_delta = 0.001
patience = 10

# declare file locations for dataset and output model
root_dir = '../dataset/semantic_drone_dataset'
img_path = root_dir + '/original_images/'
mask_path = root_dir + '/label_images_semantic/'
color_dict_loc = root_dir + "/class_dict_seg.csv"
model_file_loc = "../models/saved_unet_model.h5"

def read_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = img.astype(np.float32)
    return img

def read_mask(mask_file):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_width, img_height))
    mask = mask.astype(np.int32)
    return mask

def preprocess(input, output):
    def func(input, output):
        input = input.decode()
        output = output.decode()
        image = read_image(input)
        mask = read_mask(output)
        return image, mask

    image, mask = tf.numpy_function(func, [input, output], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes, dtype=tf.int32)
    image.set_shape([img_height, img_width, 3])  # In the Images, number of channels = 3.
    mask.set_shape([img_height, img_width, num_classes])  # In the Masks, number of channels = number of classes.
    return image, mask

def gen_tf_dataset(input, output):
    dataset = tf.data.Dataset.from_tensor_slices((input, output))  # Dataset object from Tensorflow
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(preprocess)  # Applying preprocessing to every batch in the Dataset object
    dataset = dataset.batch(batch_size)  # Determine atch-size
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch)  # Optimization
    return dataset

def preprocess_test(input):
    def func(input):
        input = input.decode()
        image = read_image(input)
        return image

    image = tf.convert_to_tensor(tf.numpy_function(func, [input], [tf.float32]))
    image = tf.reshape(image, (img_height, img_width, 3))  # In the Images, number of channels = 3.
    return image

def test_dataset(input, batch):
    dataset = tf.data.Dataset.from_tensor_slices(input)
    dataset = dataset.map(preprocess_test)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset

def train_model_and_evaluate():
    # create dataset
    file_names = list(map(lambda x: x.replace('.jpg', ''), os.listdir(img_path)))
    X_trainval, X_test = train_test_split(file_names, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=19)

    print(f"Train Size : {len(X_train)} images")
    print(f"Val Size   :  {len(X_val)} images")
    print(f"Test Size  :  {len(X_test)} images")

    y_train = X_train
    y_test = X_test
    y_val = X_val

    img_train = [os.path.join(img_path, f"{name}.jpg") for name in X_train]
    mask_train = [os.path.join(mask_path, f"{name}.png") for name in y_train]
    img_val = [os.path.join(img_path, f"{name}.jpg") for name in X_val]
    mask_val = [os.path.join(mask_path, f"{name}.png") for name in y_val]
    img_test = [os.path.join(img_path, f"{name}.jpg") for name in X_test]
    mask_test = [os.path.join(mask_path, f"{name}.png") for name in y_test]

    train_dataset = gen_tf_dataset(img_train, mask_train)
    valid_dataset = gen_tf_dataset(img_val, mask_val)
    test_ds = gen_tf_dataset(img_test, mask_test)

    train_steps = len(img_train) // batch_size
    valid_steps = len(img_val) // batch_size

    # create model
    model = multi_unet_model(n_classes=num_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=3)

    # set early stopping conditions
    es = tf.keras.callbacks.EarlyStopping(min_delta=min_delta, patience=patience)

    # perform training
    history = model.fit(train_dataset,
                        steps_per_epoch=train_steps,
                        validation_data=valid_dataset,
                        validation_steps=valid_steps,
                        epochs=num_epochs
                        )

    # save model to disk
    model.save(model_file_loc)

    # evaluate model on test dataset
    model.evaluate(test_ds, steps=14)

    # make some prediction using model
    pred = model.predict(test_dataset(img_test, batch=1), steps=40)

    # generate predictions
    predictions = np.argmax(pred, axis=3)
    label = np.array([cv2.resize(cv2.imread(mask_path + img_test[i][-7:-4] + '.png')[:, :, 0], (1200, 800)) for i in
                      range(predictions.shape[0])])
    label = label.flatten()
    predictions = predictions.flatten()

    # load color dictionary
    color_dict = pd.read_csv(color_dict_loc)

    # generate color maps for masks
    cmap = np.array(list(color_dict[[' r', ' g', ' b']].transpose().to_dict('list').values()))
    predictions = predictions.reshape(-1, img_height, img_width)
    label = label.reshape(-1, img_height, img_width)

    # display
    i = 18
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    for j in range(3):
        ax[j, 0].imshow(cmap[predictions[i + j]])
        ax[j, 1].imshow(cmap[label[i + j]])
        ax[j, 0].set_title('Prediction')
        ax[j, 1].set_title('Ground truth')
    plt.show()

if(__name__ == "__main__"):
    train_model_and_evaluate()
