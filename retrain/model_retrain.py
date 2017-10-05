import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools

import keras
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

def retrain_inception(nb_classes, layer_id, n_kernels, n_hidden):
    """
    Create Model with Inception V3 layers till the layer_id layer_id
    And then 2 convs, one hidden and one output FC layers

    Args:
        nb_classes: # of classes in putput
        layer_id  : layer name to continue from (str)

    Returns:
        new keras model with last layer

    """
    base_model = InceptionV3(weights='imagenet', include_top=True, input_shape=(150,150,3))

    while not base_model.layers[-1] == base_model.get_layer(layer_id):
        base_model.layers.pop()

    for layer in base_model.layers:
        layer.name = layer.name + 'base'

    base_out = base_model.get_layer(layer_id + 'base').output
    my_conv1 = Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu')(base_out)
    my_conv2 = Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu')(my_conv1)
    conv_out = GlobalAveragePooling2D()(my_conv2)
    hidden = Dense(n_hidden, activation='relu')(conv_out) #new FC
    predictions = Dense(nb_classes, activation='softmax')(hidden)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
