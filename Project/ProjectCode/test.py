#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2019 Created by Yiming Peng and Bing Xue
"""
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

from keras.models import model_from_json

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images - this is how you would change your model if you cannot 
    # get it to work! and then use (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype = "float") / 255.0
    y_test = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
 
 


    return X


def evaluate(X_test, y_test):
    
 # batch size is 16 for evaluation
    batch_size = 16
    """
    model = load_model('/Users/harryrodger/Desktop/MLPModels/MLPEdgeHistogram.model')
    # evaluate loaded model on test data
    model.compile(loss='categorical_crossentropy',
              optimizer='adam', #opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
              metrics=['accuracy'])
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
    print(model.summary)

    return score
    
    
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
   

    # Load Model
    # load json and create model
    json_file = open('model/Datamodelconv6.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model/")
    loaded_model.load_weights("Weights/bests_weights2epoch23loss057.hdf5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam', #opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)
              metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print(score)
    print(loaded_model.summary)
    return score



if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    X_test = preprocess_data(X_test)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    loss, accuracy = evaluate(X_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))
