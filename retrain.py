import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from data_gen import *

import keras
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

IM_WIDTH, IM_HEIGHT = 150, 150
NB_EPOCHS = 20
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt



def add_new_layer(base_model, nb_classes, layer_id):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
        layer_id  : layer name to continue from (str)
    Returns:
        new keras model with last layer
    """
    while not base_model.layers[-1] == base_model.get_layer(layer_id):
        base_model.layers.pop()

    x = base_model.get_layer(layer_id).output
    x = GlobalAveragePooling2D()(x)
    hidden = Dense(FC_SIZE, activation='relu')(x) #new FC
    predictions = Dense(nb_classes, activation='softmax')(hidden) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

def simple_model(nb_classes):
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


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=[
            metrics.categorical_accuracy,
            metrics.top_k_categorical_accuracy
            ]
        )


def train(args):
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        )

    test_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )


    base_model = InceptionV3(weights='imagenet', include_top=True)

    model = add_new_layer(base_model, nb_classes, 'mixed1')

    setup_to_transfer_learn(model, base_model)

    #model = simple_model(nb_classes)
    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto'
        )

    model.save(args.output_path  + "model.model")

    scores = model.evaluate_generator(validation_generator, steps = 500)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    np.save(args.output_path + "scores", scores, history_tl)
    print("done")


if __name__=="__main__":
    out_path = "/project/BEN_DL/output/retrained/"
    a = argparse.ArgumentParser()
    train_dir = "/project/BEN_DL/split_images/benthoz_retrain/training"
    test_dir  = "/project/BEN_DL/split_images/retrain_50/testing"
    a.add_argument("--train_dir", default = train_dir)
    a.add_argument("--val_dir", default = test_dir)
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    trial_num = max([int(d) for d in os.listdir(out_path) if os.path.isdir(out_path + d) and d.isdigit()] +[0]) + 1
    out_path = "/project/BEN_DL/output/retrained/" + str(trial_num) + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    a.add_argument("--output_path", default=out_path)
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
