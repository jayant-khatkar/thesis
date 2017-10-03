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

def dual_im(nb_classes, crop_layer, source_layer):
    """
    Create Dual Image Input Net

    Args:
        nb_classes: # of classes in putput
        crop_layer: layer of inception to go till for crops
        source_layer: " " " for sources

    Returns:
        keras Model

    """

    # get cropped image arm of net's pretrained weights
    crop_arm = InceptionV3(weights='imagenet', include_top=True, input_shape=(150,150,3))
    while not crop_arm.layers[-1] == crop_arm.get_layer(crop_layer):
        crop_arm.layers.pop()

    crop_out = crop_arm.get_layer(crop_layer).output

    #rename because layer names must be unique & using two inception inputs
    for layer in crop_arm.layers:
        layer.name = layer.name + 'crop_arm'

    # get source image arm of net's pretrained weights
    source_arm = InceptionV3(weights='imagenet', include_top=True, input_shape=(200,200,3))
    while not source_arm.layers[-1] == source_arm.get_layer(source_layer):
        source_arm.layers.pop()

    source_out = source_arm.get_layer(source_layer).output

    for layer in source_arm.layers:
        layer.name = layer.name + 'source_arm'

    #add convolutional layers with stride of 2 to reduce size
    my_conv1source = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(source_out)
    my_conv1crop   = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(crop_out)
    my_conv2source = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(my_conv1source)
    my_conv2crop   = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(my_conv1crop)

    # global pool to reduce size further
    crop_pool   = GlobalAveragePooling2D()(my_conv2crop)
    source_pool = GlobalAveragePooling2D()(my_conv2source)

    #concatenate the outputs of the two layers
    my_merge = keras.layers.concatenate([crop_pool, source_pool])

    #add hidden and output layer to produce model
    hidden = Dense(512, activation='relu')(my_merge)
    predictions = Dense(nb_classes, activation='softmax')(hidden)

    model = Model(inputs=[crop_arm.input, source_arm.input], outputs=predictions)

    return model
