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
import data_helper
import pickle
import sys
from   sklearn.metrics import confusion_matrix, f1_score
from   test      import *
from   restorer  import *

# Data parameters
embedding_size   = 400
n_words          = 120
n_classes        = 3
num_epochs       = 30
batch_size       = 200


def restore_and_train(save_path, train_path, test_path, dev_path, model_name, run_ID):
    """
    restores model saved at save_path
    trains it on data saved at train_path
    """
    sess = tf.Session()

    graph_ops = restorer(save_path, sess, 'my_ops')

    placeholders = [graph_ops[u'x_in:0'], graph_ops[u'target:0']]
    x_in   = placeholders[0]
    target = placeholders[1]

    measure_performance =[
        graph_ops[u'measure_performance/accuracy:0'],
        graph_ops[u'measure_performance/predicted_value:0'],
        graph_ops[u'measure_performance/actual_value:0']
        ]

    optimizer   = graph_ops[u'optimize/ad_opt']
    tensorboard = graph_ops[u'tensorboard_summary/MergeSummary/MergeSummary:0']
    
    
    print('\nStarting Test Performance:\n')

    test_acc, test_f1, test_cm = test_model_internal(
        test_path,
        sess,
        placeholders,
        measure_performance
        )
    display_epoch_performance("starting", test_acc, test_f1, test_cm, "test")

    tensorboard_path, op_names_path, model_path = save_paths(
        model_name,
        run_ID
        )

    ###sess.run(tf.global_variables_initializer())### do not do this, the variables get re-initialized, killing the point of the save
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(
        tensorboard_path,
        graph=tf.get_default_graph()
        )
    best_f1=0


    print('\nTraining Model...\n')
    for epoch in range(num_epochs):

        # generate batches
        batches_train = get_zipped_batches(train_path, batch_size)
        batch_index = 0

        #train on batches
    	for train_batch in batches_train:
            train_batch_xs, train_batch_ys = zip(*train_batch)
    	    _ , summary = sess.run(
                [optimizer, tensorboard],
                feed_dict={
                    x_in:   train_batch_xs,
                    target: train_batch_ys
                    }
                )

            #tensorboard summary
            summary_writer.add_summary(summary, epoch*batch_size + batch_index)
            batch_index += 1

        #calculate train & dev performance after training
        trn_acc, trn_f1, trn_cm = test_model_internal(
            train_path,
            sess,
            placeholders,
            measure_performance
            )
        dev_acc, dev_f1, dev_cm = test_model_internal(
            dev_path,
            sess,
            placeholders,
            measure_performance
            )
        display_epoch_performance(epoch, dev_acc, dev_f1, dev_cm, "dev")
        display_epoch_performance(epoch, trn_acc, trn_f1, trn_cm, "trn")

    	#if improved performance, then save model
    	if best_f1 < dev_f1:
    	    best_f1 = dev_f1
            _ = saver.save(sess, model_path)

    #test acter completing training
    print('\nCompleted, test performance:\n')
    test_acc, test_f1, test_cm = test_model_internal(
        test_path,
        sess,
        placeholders,
        measure_performance
        )
    display_epoch_performance("final", test_acc, test_f1, test_cm, "test")


if __name__ == '__main__':
    model_name   =  'GRU_net'
	run_ID = 15
    save_path    =  'GRU_net/14/model_save/'
    local_path   = "/mnt/resource/ML/Amit/sentiment-experiment/data/"
    train_path   = local_path + "sentiment-train-1-we_fg-cnn-120.txt"
    dev_path     = local_path + "sentiment-dev-1-we_fg-cnn-120.txt"
    test_path    = local_path + "sentiment-test-16-we_fg-cnn-120.txt"
    
    restore_and_train(save_path, train_path, test_path, dev_path, model_name, run_ID)
