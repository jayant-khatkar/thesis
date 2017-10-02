"""
Test model
"""

import tensorflow as tf
import numpy as np
#import data_helper
import os
import sys
model_locations = os.path.abspath(os.path.join(os.path.curdir, '..', 'Models'))
sys.path.insert(0, model_locations)
other_locations = os.path.abspath(os.path.join(os.path.curdir, '..'))
sys.path.insert(0, other_locations)
from   sklearn.metrics import confusion_matrix, f1_score
from data_gen import *


def test_model_internal(test_path, sess, placeholders, measure_performance, out_dim):
    """
    Reads data at file_path

    Internal function is separate so it can be accessed by restored model (when the class is not known)

    Returns accuracy, average f1 score and confusion matrix
    """
    test_batches = batch_image_generator(test_path, 200, 1)

    accuracy  = np.array([])
    y_pred    = np.array([])
    y_real    = np.array([])

    for test_batch_xs, test_batch_ys in test_batches:

        accuracy_batch, y_pred_batch, y_real_batch = sess.run(
            measure_performance,
            feed_dict = {
                placeholders[0]:   test_batch_xs,
                placeholders[1]:   test_batch_ys
                }
            )

        y_pred   = np.append(y_pred,   y_pred_batch)
        y_real   = np.append(y_real,   y_real_batch)
        accuracy = np.append(accuracy, accuracy_batch)

    accuracy = accuracy.mean()
    labels = np.arange(out_dim) 
    f1_sc = f1_score(y_real, y_pred, average=None, labels = labels)

    cm = confusion_matrix(y_real, y_pred, labels)

    return accuracy, cm, y_pred, y_real, labels

def test_model(test_path, model, sess, out_dim):
    """
    wrapper for the internal function so it can easily accessed when the model is given

    Returns accuracy, average f1 score and confusion matrix
    """
    placeholders = [model.x_in, model.target]
    return test_model_internal(test_path, sess, placeholders, model.measure_performance,out_dim)

def display_epoch_performance(epoch, accuracy, confusion_matrix, source):
    print('At Epoch: '+ str(epoch))
    print('    ' + source + '    accuracy: '   + str(accuracy))
    print('    ' + source + ' conf matrix: \n' + str(confusion_matrix) +'\n')


def save_paths(save_path, model_name):
    path = os.path.join(save_path, model_name)
    run_ID = max(map(int,os.listdir(path)))+1
    path = os.path.join(path, str(run_ID))
    tensorboard = os.path.join(path, "tensorboard_log")
    model_save  = os.path.join(path, "model_save")


    if not os.path.exists(tensorboard):
        os.makedirs(tensorboard)

    if not os.path.exists(model_save):
        os.makedirs(model_save)

    model_save  = os.path.join(model_save,"model.ckpt")

    return tensorboard, model_save
