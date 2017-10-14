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
from keras.models import load_model

FC_SIZE = 1024


def dual_im(nb_classes, source_layer, n_kernels, n_hidden):
    """
    Create Dual Image Input Net

    Args:
        nb_classes: # of classes in putput
        source_layer: layer of inception to go till for sources

    Returns:
        keras Model

    """

    # get cropped image arm of net's pretrained weights
    crop_arm = load_model("/project/BEN_DL/output/retrained/finetune/6/model.h5")
    # crop_arm = InceptionV3(weights='imagenet', include_top=True, input_shape=(150,150,3))
    while not crop_arm.layers[-1].name == "global_average_pooling2d_1":
        crop_arm.layers.pop()

    crop_out = crop_arm.layers[-1].output

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
    my_conv1source = Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu')(source_out)
    # my_conv1crop   = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(crop_out)
    my_conv2source = Conv2D(n_kernels, (3,3), strides=(1, 1), padding='valid', activation='relu')(my_conv1source)
    # my_conv2crop   = Conv2D(64, (3,3), strides=(2, 2), padding='valid', activation='relu')(my_conv1crop)

    # global pool to reduce size further
    # crop_pool   = GlobalAveragePooling2D()(my_conv2crop)
    source_pool = GlobalAveragePooling2D()(my_conv2source)

    #concatenate the outputs of the two layers
    my_merge = keras.layers.concatenate([crop_out, source_pool])

    #add hidden and output layer to produce model
    hidden = Dense(n_hidden, activation='relu')(my_merge)
    predictions = Dense(nb_classes, activation='softmax')(hidden)

    model = Model(inputs=[crop_arm.input, source_arm.input], outputs=predictions)

    return model
