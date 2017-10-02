"""
LSTM Deep-RNN Model

mulilayer LSTM cells, no fully connected hidden layer

Author: Jayant Khatkar
"""

import tensorflow as tf
import numpy as np
import os
from scope_decorator import *

class Deep_RNN:

    def __init__(self, x_in, target, embed_len, n_words, n_classes, hyperPs):
        """
        x_in and target must be of type tf.placeholder
        remaining arguments are int data parameters
        """
        self.x_in = x_in
        self.x = tf.reshape(self.x_in, [-1,n_words,embed_len])
        self.y = target

        self.n_classes = n_classes
        self.n_words = n_words
        self.embed_len = embed_len
        self.model_name = "Deep_RNN"

        self.load_hyperparameters(hyperPs)
        self.init_weights
        self.prediction
        self.optimize
        self.measure_performance


    def load_hyperparameters(self, hyperPs):
        self.learning_rate    = hyperPs[0]
        self.input_keep_prob  = hyperPs[1]
        self.output_keep_prob = hyperPs[1]
        self.hidden_LSTM      = hyperPs[2]
        self.num_layers       = hyperPs[3]
        self.words_used       = hyperPs[4]


    @define_scope
    def init_weights(self):

        self.weights = {
            'layer1': tf.get_variable(
                name        = "W_1",
                shape       = [self.hidden_LSTM, self.n_classes],
                initializer = tf.contrib.layers.xavier_initializer()
                )
        }

        self.biases = {
            'layer1': tf.Variable(
                tf.random_normal([self.n_classes])
                )
        }


    @define_scope
    def prediction(self):
        #define LSTM cell and wrappers
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            self.hidden_LSTM,
            state_is_tuple = True
            )

        multicell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * self.num_layers,
            state_is_tuple = True
            )

        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
    		multicell,
    		input_keep_prob  = self.input_keep_prob,
    		output_keep_prob = self.output_keep_prob
    		)

        #unroll RNN
        outputs, states = tf.nn.dynamic_rnn(
            dropout_cell, self.x, dtype=tf.float32)

        predict = tf.matmul(
            outputs[:,self.words_used-1,:],
            self.weights['layer1']
            ) + self.biases['layer1']

        return predict


    @define_scope
    def optimize(self):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            self.prediction,
            self.y
            )

        self.cost = tf.reduce_mean(cross_entropy)

        adam_op = tf.train.AdamOptimizer()
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
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name = "accuracy")

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
        print("Embedding size:    " + str(self.embed_len))
        print("hidden_LSTM:       " + str(self.hidden_LSTM))
        print("num_layers:        " + str(self.num_layers))
        print("Input keep prob:   " + str(self.input_keep_prob))
        print("Output keep prob:  " + str(self.output_keep_prob))
