import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools
import keras

from dual_im_gen import dual_im_gen
from DI_model import dual_im

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



batch_size = 32
NB_EPOCHS  = 20

test_address = "/project/BEN_DL/split_images/benthoz_retrain/testing"
train_address = "/project/BEN_DL/split_images/benthoz_retrain/training"
out_path = "/project/BEN_DL/output/DI/"

nb_classes = len(glob.glob(train_address + "/*"))

def transfer_learn_DI(model):

    for layer in model.layers:
        if layer.name.endswith('source_arm') or layer.name.endswith('crop_arm'):
            layer.trainable = False

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=[
            metrics.categorical_accuracy,
            metrics.top_k_categorical_accuracy,
            #top3_acc
            ]
        )

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
    cnt = len(glob.glob(os.path.join(directory,'*/*.jpg')))
    return cnt

if __name__=="__main__":
    trial_num = max([int(d) for d in os.listdir(out_path) if os.path.isdir(out_path + d) and d.isdigit()] +[0]) + 1
    save_dir = out_path + str(trial_num) + "/"

    nb_train_samples = get_nb_files(train_address)
    nb_val_samples = get_nb_files(test_address)

    train_steps = int(nb_train_samples/batch_size)
    val_steps = int(nb_val_samples/batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = dual_im(nb_classes, 'mixed0', 'mixed2')

    transfer_learn_DI(model)

    dim_gen_test  = dual_im_gen(test_address,  batch_size)
    dim_gen_train = dual_im_gen(train_address, batch_size)

    # tensorboard = keras.callbacks.TensorBoard(
    #     log_dir=save_dir,
    #     histogram_freq=5,
    #     batch_size=batch_size,
    #     write_graph=True,
    #     write_grads=False,
    #     write_images=True
    #     )

    history_tl = model.fit_generator(
        dim_gen_train,
        epochs=NB_EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=dim_gen_test,
        validation_steps=val_steps,
        class_weight='auto',
#        callbacks = [tensorboard], #tensorboard only works when not using generator for validation data if printing histograms
        verbose =1
        )

    model.save(save_dir  + "model.model")

    scores = model.evaluate_generator(dim_gen_test, steps = val_steps)
    y_pred, y_true = predictor(model, dim_gen_test, steps = val_steps)

    conf = confusion_matrix(y_true = y_true, y_pred = y_pred)
    f1scores  = f1_score(y_true = y_true, y_pred = y_pred, average = None)

    np.savez(save_dir + "output_vars.npz",
        scores      = scores,
        hist        = history_tl.history,
        y_pred      = y_pred,
        y_true      = y_true,
        conf        = conf,
        f1scores    = f1scores
        )
