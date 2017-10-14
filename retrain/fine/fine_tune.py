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

IM_WIDTH, IM_HEIGHT = 150, 150
NB_EPOCHS = 10
BAT_SIZE = 32

def predictor(model, test_generator, steps):
    y_pred = np.array([]).reshape(0,24)
    y_true = np.array([]).reshape(0,24)

    for i in range(steps):
        features, labels = next(test_generator)
        batch_pred = model.predict(features)

        y_pred = np.append(y_pred, batch_pred, axis=0)
        y_true = np.append(y_true, labels, axis=0)

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred, y_true

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_finetune(model, freeze_layer):
    """Freeze all layers and compile the model"""
    flag=False
    for layer in model.layers:

        layer.trainable = flag
        if layer.name == freeze_layer:
            flag=True

#    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    model.compile(
        optimizer= Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=[
            metrics.categorical_accuracy,
            metrics.top_k_categorical_accuracy,
#            top3_acc
            ]
        )


def train(args):
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_test_samples = get_nb_files(args.test_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    train_steps = int(nb_train_samples/batch_size)
    val_steps = int(nb_val_samples/batch_size)
    test_steps = int(nb_test_samples/batch_size)

    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        )

    val_datagen =  ImageDataGenerator(
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

    validation_generator = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )


    model = load_model(args.model_dir)



    setup_to_finetune(model, 'mixed2base')

    history_tl1 = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        class_weight='auto',
#        callbacks = [tensorboard],
        verbose =2
        )

    print("Done 1/3 rounds of fine tuning")

    setup_to_finetune(model, 'mixed1base')
    history_tl2 = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        class_weight='auto',
#        callbacks = [tensorboard],
        verbose =2
        )
    print("Done 2/3 rounds of fine tuning")

    setup_to_finetune(model, 'mixed0base')
    history_tl3 = model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        class_weight='auto',
#        callbacks = [tensorboard],
        verbose =2
        )
    print("Done 3/3 rounds of fine tuning")

    model.save(args.output_path  + "model.h5")

    scores = model.evaluate_generator(test_generator, steps = test_steps)
    y_pred, y_true = predictor(model, test_generator, steps = test_steps)

    conf = confusion_matrix(y_true = y_true, y_pred = y_pred)
    f1scores  = f1_score(y_true = y_true, y_pred = y_pred, average = None)

    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("F1-score: %.2f%%" % (np.mean(f1scores)*100))

    np.savez(args.output_path + "output_vars.npz",
        scores      = scores,
        hist1        = history_tl1.history,
        hist2        = history_tl2.history,
        hist3        = history_tl3.history,
        y_pred      = y_pred,
        y_true      = y_true,
        conf        = conf,
        f1scores    = f1scores
        )


if __name__=="__main__":
    out_path = "/project/BEN_DL/output/retrained/finetune/"
    a = argparse.ArgumentParser()
    train_dir = "/project/BEN_DL/split_images/benthoz_retrain/training"
    test_dir = "/project/BEN_DL/split_images/benthoz_retrain/testing"
    val_dir  = "/project/BEN_DL/split_images/retrain_50/testing"
    model_dir = "/project/BEN_DL/output/retrained/17/model.model"
    a.add_argument("--train_dir", default = train_dir)
    a.add_argument("--test_dir", default = test_dir)
    a.add_argument("--val_dir", default = val_dir)
    a.add_argument("--model_dir", default = model_dir)
    a.add_argument("--nb_epoch", default=NB_EPOCHS, type=int)
    a.add_argument("--batch_size", default=BAT_SIZE, type=int)
    trial_num = max([int(d) for d in os.listdir(out_path) if os.path.isdir(out_path + d) and d.isdigit()] +[0]) + 1
    save_dir = out_path + str(trial_num) + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    a.add_argument("--output_path", default=save_dir)
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
