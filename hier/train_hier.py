import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools
import keras

from hier_gen import hier_gen
from hier_model import hier

from keras.models import Model
from keras import metrics
from sklearn.metrics import confusion_matrix, f1_score
from keras.utils import plot_model

batch_size = 32
NB_EPOCHS  = 10

test_address  = "/project/BEN_DL/split_images/benthoz_retrain/testing"
#test_address = "/Users/user1/Desktop/thesis/data/split_images/benthoz_retrain/testing"
val_address   = "/project/BEN_DL/split_images/retrain_50/testing"
#val_address = test_address
train_address = "/project/BEN_DL/split_images/benthoz_retrain/training"
#train_address = "/Users/user1/Desktop/thesis/data/split_images/benthoz_retrain/training"
out_path      = "/project/BEN_DL/output/hier/"
#out_path = "/Users/user1/Desktop/pbs_outputs"
def transfer_learn_hier(model):

    for layer in model.layers:
        if layer.name.endswith('inception'):
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

    y0_pred = np.array([]).reshape(0,3)
    y0_true = np.array([]).reshape(0,3)

    y1_pred = np.array([]).reshape(0,8)
    y1_true = np.array([]).reshape(0,8)

    y2_pred = np.array([]).reshape(0,19)
    y2_true = np.array([]).reshape(0,19)

    for i in range(steps):
        features, labels = next(test_generator)

        batch0t, batch1t, batch2t = labels
        batch0p, batch1p, batch2p = model.predict(features)

        y0_pred = np.append(y0_pred, batch0p, axis=0)
        y1_pred = np.append(y1_pred, batch1p, axis=0)
        y2_pred = np.append(y2_pred, batch2p, axis=0)

        y0_true = np.append(y0_true, batch0t, axis=0)
        y1_true = np.append(y1_true, batch1t, axis=0)
        y2_true = np.append(y2_true, batch2t, axis=0)

    y0_true = np.argmax(y0_true, axis=1)
    y1_true = np.argmax(y1_true, axis=1)
    y2_true = np.argmax(y2_true, axis=1)

    y0_pred = np.argmax(y0_pred, axis=1)
    y1_pred = np.argmax(y1_pred, axis=1)
    y2_pred = np.argmax(y2_pred, axis=1)

    return [y0_pred, y1_pred, y2_pred], [y0_true, y1_true, y2_true]

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    cnt = len(glob.glob(os.path.join(directory,'*/*.jpg')))
    return cnt

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--layer", default='mixed2')
    a.add_argument("--n_kernels", default=64, type=int)
    a.add_argument("--n_hidden", default=512, type=int)
    args = a.parse_args()

    trial_num = max([int(d) for d in os.listdir(out_path) if os.path.isdir(out_path + d) and d.isdigit()] +[0]) + 1
    save_dir = out_path + str(trial_num) + "/"

    nb_train_samples = get_nb_files(train_address)
    nb_val_samples = get_nb_files(val_address)
    nb_test_samples = get_nb_files(test_address)

    train_steps = int(nb_train_samples/batch_size)
    val_steps = int(nb_val_samples/batch_size)
    test_steps = int(nb_test_samples/batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = hier(args.layer, args.n_kernels, args.n_hidden)

    transfer_learn_hier(model)

    gen_test  = hier_gen(test_address,  batch_size)
    gen_val   = hier_gen(val_address,   batch_size)
    gen_train = hier_gen(train_address, batch_size)

    # tensorboard = keras.callbacks.TensorBoard(
    #     log_dir=save_dir,
    #     histogram_freq=5,
    #     batch_size=batch_size,
    #     write_graph=True,
    #     write_grads=False,
    #     write_images=True
    #     )

    history_tl = model.fit_generator(
        gen_train,
        epochs=NB_EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=gen_val,
        validation_steps=val_steps,
        # class_weight='auto',
#        callbacks = [tensorboard], #tensorboard only works when not using generator for validation data if printing histograms
        verbose =1
        )

    model.save(save_dir  + "model.model")


    #### NOT REALLY SURE WHAT THIS DOES WITH MULTIPLE y's
    scores = model.evaluate_generator(gen_test, steps = test_steps)

    ### TO DO: FIX THIS FOR MULTIPLE y's
    y_pred, y_true = predictor(model, gen_test, steps = test_steps)

    conf=[]
    f1scores=[]
    for y_p, y_t in zip(y_pred, y_true):
        conf.append(confusion_matrix(y_true = y_t, y_pred = y_p))
        f1scores.append(f1_score(y_true = y_t, y_pred = y_p, average = None))

    np.savez(save_dir + "output_vars.npz",
        scores      = scores,
        hist        = history_tl.history,
        y_pred      = y_pred,
        y_true      = y_true,
        conf        = conf,
        f1scores    = f1scores
        )
