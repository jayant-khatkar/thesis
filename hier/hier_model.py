import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def hier(my_layer, n_kernels, n_hidden):
    """
    Create Hierarchical Output Net
    Number of layers/ classes per layer are hard coded

    Args:
        my_layer: layer of inception to go till

    Returns:
        keras Model

    """

    # get cropped image arm of net's pretrained weights
    inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(150,150,3))
    while not inception.layers[-1] == inception.get_layer(my_layer):
        inception.layers.pop()

    inception_out = inception.get_layer(my_layer).output

    #rename because layer names must be unique & using two inception inputs
    for layer in inception.layers:
        layer.name = layer.name + 'inception'

    #add convolutional layers with stride of 2 to reduce size
    my_conv1   = Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu')(inception_out)
    my_conv2   = Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu')(my_conv1)

    # global pool to reduce size further
    pool   = GlobalAveragePooling2D()(my_conv2)

    #add hidden and output layer to produce model
    hidden = Dense(n_hidden, activation='relu')(pool)

    y0 = Dense(3, activation='softmax')(hidden)
    y1 = Dense(8, activation='softmax')(hidden)
    y2 = Dense(19, activation='softmax')(hidden)

    model = Model(inputs=inception.input, outputs=[y0,y1,y2])

    return model
