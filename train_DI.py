import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools
import keras

from dual_im_gen import dual_im_gen
from models import dual_im

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


nb_classes = 24
batch_size = 32

test_address = "/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/testing"
train_address = "/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/training"
out_path = "/project/BEN_DL/output/DI/"

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


if __name__=="__main__":
    trial_num = max([int(d) for d in os.listdir(out_path) if os.path.isdir(out_path + d) and d.isdigit()] +[0]) + 1
    save_dir = out_path + str(trial_num) + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    transfer_learn_DI(model)

    dim_gen_test  = dual_im_gen(test_address,  batch_size)
    dim_gen_train = dual_im_gen(train_address, batch_size)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=save_dir,
        histogram_freq=5,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=True
        )

    history_tl = model.fit_generator(
        dim_gen_train,
        epochs=1,
        steps_per_epoch=2,
        validation_data=dim_gen_test,
        validation_steps=1,
        class_weight='auto',
#        callbacks = [tensorboard], #tensorboard only works when not using generator for validation data if printing histograms
        verbose =2
        )

    model.save(save_dir  + "model.model")
