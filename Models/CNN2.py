"""
Basic CNN for benthoz classification

Author: Jayant Khatkar
"""

import tensorflow as tf
import numpy as np
import os
from scope_decorator import *

class CNN2:

    def __init__(self, x_in, target, im_size, n_classes):#, hyperPs):
        """
        x_in and target must be of type tf.placeholder
        remaining arguments are int data parameters
        """
        self.x_in = x_in
        self.target = target
        self.y = tf.one_hot(target,n_classes)

        self.n_classes = n_classes
        self.im_size = im_size
        self.model_name = "CNN2"

        self.load_hyperparameters()#hyperPs)
        self.init_weights
        self.prediction
        self.optimize
        self.measure_performance


    def load_hyperparameters(self):#, hyperPs):
        self.learning_rate    = 0.001#hyperPs[0]
        self.drop_keep_prob   = 0.9#hyperPs[1]
        #self.n_channels       = hyperPs[2]
        #self.filter_sizes     = hyperPs[3]
        #self.hidden_layer     = hyperPs[4]


    @define_scope
    def init_weights(self):

        self.weights = {
            # 3x3 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
            # 3x3 conv, 32 inputs, 32 outputs
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
	        'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32])),
	        'wc4': tf.Variable(tf.random_normal([3, 3, 32, 32])),

	        'wd1': tf.Variable(tf.random_normal([32*self.im_size**2, 1024])),
            # 1024 inputs, n_classes outputs 
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([32])),
            'bc3': tf.Variable(tf.random_normal([32])),
            'bc4': tf.Variable(tf.random_normal([32])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }


    @define_scope
    def prediction(self):

        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(
                x, 
                W, 
                strides=[1, strides, strides, 1], 
                padding='SAME', 
                )
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)


        def maxpool2d(x, k=2):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                  padding='SAME')

        # Convolution Layer
        conv1 = conv2d(self.x_in, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        #conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        #conv2 = maxpool2d(conv2, k=2)

        #hidden = tf.nn.relu(tf.matmul(conv_out, self.weights['dense1']) + self.biases['dense1'])
        conv3 = conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
        conv4 = conv2d(conv3, self.weights['wc4'], self.biases['bc4'])

        fc1 = tf.contrib.layers.flatten(conv4)
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.drop_keep_prob)

        predict = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        #tf.nn.softmax(tf.matmul(hidden, self.weights['dense2']) + self.biases['dense2'])

        return predict


    @define_scope
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction,
            labels=self.y
            )

        self.cost = tf.reduce_mean(cross_entropy)
        adam_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = adam_op.minimize(loss = self.cost, name = "ad_opt")
        return self.optimizer


    @define_scope
    def measure_performance(self):

        self.model_pred = tf.argmax(
            self.prediction,
            1,
            name = "predicted_value"
            )

        self.real_val = tf.argmax(self.y, 1, name = "actual_value")
        correct  = tf.equal(self.model_pred, self.real_val)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")

        #return model prediction and real_val as well to caluclate f1 score
        #outside of tensorflow
        return [self.accuracy, self.model_pred, self.real_val]


    @define_scope
    def tensorboard_summary(self):

        tf.summary.scalar("Accuracy", self.accuracy)
        tf.summary.scalar("Loss", self.cost)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        #for grad, var in grads:
        #    tf.histogram_summary(var.name + '/gradient', grad)

        self.merged_summary_op = tf.summary.merge_all()

        return self.merged_summary_op


    def info(self):

        print("Model Name:        " + self.model_name)
        #print("filter sizes:      " + str(self.filter_sizes))
        #print("n_channels:        " + str(self.n_channels))
        #print("Dropout keep prob: " + str(self.drop_keep_prob))
