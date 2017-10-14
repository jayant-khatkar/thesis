import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools

import keras

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras import metrics
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from keras.models import load_model
from scipy.misc import imread, imshow, imresize, imsave
from shutil import copyfile

val_dir = "/project/BEN_DL/split_images/retrain_50/testing"
model_dir = "/project/BEN_DL/output/retrained/finetune/6/model.h5"
out_dir = "/project/BEN_DL/split_images/errors/"

model = load_model(model_dir)

crop_addresses = glob.glob(os.path.join(data_dir,'*/*.jpg'))

for im_dir in crop_addresses:
    im = imresize(imread(im_dir), (150,150)).reshape((1, 150, 150, 3))
    true = int(im_dir.split('/')[-2])
    pred = np.argmax(model.predict(im), axis=1)[0]

    if not true==pred:
        if not os.path.isdir(os.path.join(out_dir,str(true))):
            os.mkdir(os.path.join(out_dir,str(true)))

        if not os.path.isdir(os.path.join(out_dir,str(true), str(pred))):
            os.mkdir(os.path.join(out_dir,str(true), str(pred)))

        im_cp_dir = os.path.join(out_dir, str(true), str(pred), )
        imsave(im_dir, im)


def predictor(model, test_generator, steps):
    y_pred = np.array([]).reshape(0,24)
    y_true = np.array([]).reshape(0,24)

    for i in range(steps):
        features, labels = next(test_generator)
        batch_pred = model.predict(features, verbose=1)

        y_pred = np.append(y_pred, batch_pred, axis=0)
        y_true = np.append(y_true, labels, axis=0)

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred, y_true

test_datagen =  ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    )
test_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=50,
    )

y_true, y_pred = predictor(model, test_generator, 24)
print(np.sum(y_true==y_pred)/len(y_true))
