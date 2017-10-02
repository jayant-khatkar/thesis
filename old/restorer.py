'''
Restore saved models

Functions relating to restoring saved models

restoring saved models
testing restored models
training restored models


Jayant Khatkar
'''

import tensorflow as tf
import numpy as np
import os
#import data_helper
import pickle
from   sklearn.metrics import confusion_matrix, f1_score
from   test import *


# Data parameters
embedding_size   = 400
n_words          = 120
n_classes        = 3
num_epochs       = 1
batch_size       = 200


def restorer_old(save_path, sess):
    """
    deprecated restorer, similar to the new
    restorer, but loads graph_ops from the file, instead of tf collection
    the retured dict is slightly different: dict(name) = name, instead of
    dict(name) = tf_object, but these can be used interchangeably
    """
    restorer = tf.train.import_meta_graph(
        save_path+'model.ckpt.meta'
        )

    restorer.restore(
        sess,
        tf.train.latest_checkpoint(save_path)
        )

    graph_ops = {}

    with open (save_path + '../op_names.txt', 'rb') as fp:
        itemlist=pickle.load(fp)

    for i in itemlist:
        graph_ops[i] = i

    return graph_ops


def restorer(save_path, sess, grap_op_collections):
    """
	inputs:
		- savepath is a string to the location of both the metagraph and the save data
		- sess is the current tf.session (needed to restore save data)
		- grap_op_collections is a list of strings, which are the names of the collections to be loaded

    Restores graph and save data model saved in save_path into sess

	Returns:
		- dict of graph_ops stored in the desired collections (hashed by op names)
    """

    restorer = tf.train.import_meta_graph(
        save_path+'model.ckpt.meta'
        )

    restorer.restore(
        sess,
        tf.train.latest_checkpoint(save_path)
        )

    graph_ops ={}

    for collection in grap_op_collections:
	    for op in tf.get_collection(collection):
		    graph_ops[op.name] = op

    return graph_ops


def testing_restorer():
    sess = tf.Session()
    save_path =  'GRU_net/13/model_save/'
    col = ['m']

    graph_ops=restorer_old(save_path, sess)

    print(graph_ops.keys())

    return


def restore_and_test(save_path, test_path):
    """
    restores model stored at save_path
    tests model on data saved at test_path
    """
    sess = tf.Session()

    graph_ops = restorer(save_path, sess, 'my_ops')

    #this step requires knowledge of the names of saved graph_ops
    placeholders = [graph_ops[u'x_in:0'], graph_ops[u'target:0']]
    measure_performance =[
        graph_ops[u'measure_performance/accuracy:0'],
        graph_ops[u'measure_performance/predicted_value:0'],
        graph_ops[u'measure_performance/actual_value:0']
        ]
    print("model restored, testing...")

    test_acc, test_f1, test_cm = test_model_internal(
        test_path,
        sess,
        placeholders,
        measure_performance
        )
    display_epoch_performance("saved", test_acc, test_f1, test_cm, "test")


def print_restored_ops(collection):
    """
	prints the names of all the ops saved in a given collection
	"""

    for op in tf.get_collection(collection):
        print(op.name)

    return

if __name__ == '__main__':

    save_path =  'biRNN/2/model_save/'
    local_path       = "/path/to/data/"
    train_path       = local_path + "train_set.txt"
    dev_path         = local_path + "dev_set.txt"
    test_path        = local_path + "test_set.txt"
    restore_and_test(save_path, test_path)
