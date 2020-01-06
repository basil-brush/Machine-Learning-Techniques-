#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import keras
from keras.models import model_from_json

from datetime import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    #Building the CNN
    
    #1 Initialise CNN kernel_regularizer
    model = Sequential()
    #1 Convolutional Layer
    #A convolutional layer contains a set of filters whose parameters need to be learned. 
    # The height and weight of the filters are smaller than those of the input volume. 
    model.add(Conv2D(32,kernel_size = 3,padding='same', activation = 'relu', input_shape = (64,64,3), kernel_regularizer = regularizers.l2(1e-4))) 
    model.add(BatchNormalization())
    #parameters of CONV2D
    #filtersize: This is the size of the output dimension (i.e. the number of output filters in the convolution)
    #kernel sise: This specifies the height and width of the 2D convolution window.
    #activation: We select an activation function also called non-linearity to be used by our neural network.
    #input shape: size of the image and its channels - 3 for RGB array 
    #2 Layer
    model.add(Conv2D(32,kernel_size = 3,padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)))
    #L2 regularizer is added into the loss function as a penalty term to be minimized for all weights and neurons in the network.
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #3 Layer
    model.add(Conv2D(64,kernel_size = 3,padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #4 Layer
    model.add(Conv2D(64,kernel_size = 3,padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    #5 Layer
    model.add(Conv2D(128,kernel_size = 3,padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #6 Layer
    model.add(Conv2D(128,kernel_size = 3,padding='same', activation = 'relu', kernel_regularizer = regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    #7 Flattening
    model.add(Flatten())
    #combats overfitting Dropout randomly drops some layers in a neural networks and then learns with the reduced network. This way, the network learns to be independent and not reliable on a single layer. 
    #Bottom-line is that it helps in overfitting. 0.5 means to randomly drop half of the layers.
    model.add(Dense(units=3, activation='softmax'))
    #A convolutional layer is much more specialized, and efficient, than a fully connected layer. In a fully connected 
    # layer each neuron is connected to every neuron in the previous layer, and each connection has it's own weight.
    #8 Compilation of CNN
    model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

    
    print(model.summary())
    print('Initialise and Compilation of CNN complete')

    
    
    return model


def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here

    #Preprocessing 
    # Data augmentation - creation of more images to train on
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
		height_shift_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory( #how can I check if I am actually making many more instances of pictures.
        '/Users/harryrodger/Desktop/data',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory( #how can I check if I am actually making many more instances of pictures.
        '/Users/harryrodger/Desktop/ProjectCOMP309/ProjectCode/data/test',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical'
    )

    print('Data augmentation complete')
    weights = "/Users/harryrodger/Desktop/ProjectCOMP309/ProjectCode/Weights/bests_weights2:epoch:{epoch:02d}loss:{val_loss:.2f}.hdf5"
    checkpoint_weights = ModelCheckpoint(filepath=weights, monitor='val_loss',verbose=1, save_best_only=True, mode ='min')

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model.fit_generator( 
        training_set,
        steps_per_epoch=2000, 
        epochs = 50,
        validation_data=test_set,
        validation_steps = 15,
        callbacks=[checkpoint_weights,tensorboard_callback])

    return model 


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    # serialize model to JSON
    model_json = model.to_json()
    with open("model/Datamodelconv6.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/Datamodel3onv6.h5")
    print("Saved model to disk")
    """
    model.save("model/model1.h5")
    """
    print("Model Saved Successfully.")


if __name__ == '__main__': #run the code and show them the warnings you get, are there any which I should be worried about? / can you show me how to use google colab
    model = construct_model()
    model = train_model(model)
    save_model(model)


#automatically remove outliers - histograms
#MLP - use weka to extract the features  for the MLP
#CNN

