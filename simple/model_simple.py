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

def simple_model(nb_classes, n_kernels, n_hidden):
    """
    Create Simple CNN - 5conv layers, 1 FC hidden, 1 FC output

    Args:
        nb_classes: # of classes in putput

    Returns:
        keras Model

    """
    model = Sequential()
    model.add(Conv2D(n_kernels, (3,3), strides=(2,2), activation='relu',input_shape=(100,100,3)))
    model.add(Conv2D(n_kernels, (3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(n_kernels, (3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(n_kernels, (3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(n_kernels, (3,3), strides=(2,2), padding='valid', activation='relu'))
    #model.add(Conv2D(n_kernels, (3,3), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=[
            metrics.categorical_accuracy,
            metrics.top_k_categorical_accuracy,
#            top3_acc
            ]
        )
    return model
