import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools

import keras
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import metrics
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model

FC_SIZE = 1024

def retrain_inception(nb_classes, layer_id):
    """
    Create Model with Inception V3 layers till the layer_id layer_id
    And then one hidden and one output FC layers

    Args:
        nb_classes: # of classes in putput
        layer_id  : layer name to continue from (str)

    Returns:
        new keras model with last layer

    """
    base_model = InceptionV3(weights='imagenet', include_top=True)

    while not base_model.layers[-1] == base_model.get_layer(layer_id):
        base_model.layers.pop()

    x = base_model.get_layer(layer_id).output
    x = GlobalAveragePooling2D()(x)
    hidden = Dense(FC_SIZE, activation='relu')(x) #new FC
    predictions = Dense(nb_classes, activation='softmax')(hidden)
    model = Model(inputs=base_model.input, outputs=predictions)
    return base_model, model

def simple_model(nb_classes):
    """
    Create Simple CNN - 2conv layers, 1 FC hidden, 1 FC output

    Args:
        nb_classes: # of classes in putput

    Returns:
        keras Model

    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(150,150,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])
    return model
