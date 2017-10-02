"""
bidirectional RNN using GRU cells with fully connectedModel

bidirectional version of GRU_net

Author: Jayant Khatkar
"""

import tensorflow as tf
import numpy as np
import os
from scope_decorator import *

class biRNN:

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
        self.model_name = "biRNN"

        self.load_hyperparameters(hyperPs)
        self.init_weights
        self.prediction
        self.optimize
        self.measure_performance
        self.tensorboard_summary
        self.save_graph_ops



    def load_hyperparameters(self, hyperPs):
        self.learning_rate    = hyperPs[0]
        self.input_keep_prob  = hyperPs[1]
        self.output_keep_prob = hyperPs[1]
        self.num_GRU_units    = hyperPs[2]
        self.num_layers       = hyperPs[3]
        self.words_used       = hyperPs[4]
        self.hidden_layer     = hyperPs[5]


    @define_scope
    def init_weights(self):

        self.weights = {
            'layer1': tf.get_variable(
                name        = "W_1",
                shape       = [self.num_GRU_units*2, self.hidden_layer],
                initializer = tf.contrib.layers.xavier_initializer()
                ),
            'layer2': tf.get_variable(
                name        = "W_2",
                shape       = [self.hidden_layer, self.n_classes],
                initializer = tf.contrib.layers.xavier_initializer()
                )
        }

        self.biases = {
            'layer1': tf.Variable(
                tf.random_normal([self.hidden_layer])
                ),
            'layer2': tf.Variable(
                tf.random_normal([self.n_classes])
                )
        }


    @define_scope
    def prediction(self):
        #define GRU cell and wrappers
        gru_cell_1 = tf.nn.rnn_cell.GRUCell(
            self.num_GRU_units
            )
			
        gru_cell_2 = tf.nn.rnn_cell.GRUCell(
            self.num_GRU_units
            )
		
        multicell_1 = tf.nn.rnn_cell.MultiRNNCell(
            [gru_cell_1] * self.num_layers,
            state_is_tuple = True
            )

        multicell_2 = tf.nn.rnn_cell.MultiRNNCell(
            [gru_cell_2] * self.num_layers,
            state_is_tuple = True
            )

        dropout_cell_1 = tf.nn.rnn_cell.DropoutWrapper(
            multicell_1,
            input_keep_prob  = self.input_keep_prob,
            output_keep_prob = self.output_keep_prob
            )
		
        dropout_cell_2 = tf.nn.rnn_cell.DropoutWrapper(
            multicell_2,
            input_keep_prob  = self.input_keep_prob,
            output_keep_prob = self.output_keep_prob
            )

        #unroll RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            gru_cell_1,
            gru_cell_2,
            self.x,
            dtype=tf.float32,
            sequence_length=tf.fill(tf.shape(self.x[:,1,1]), self.n_words)
            )

        fw_out, bw_out = outputs

        RNN_out = tf.concat(1,[fw_out[:,self.words_used-1,:], bw_out[:,self.words_used-1,:]])
        
        hidden = tf.matmul(
            RNN_out,
            self.weights['layer1']
            ) + self.biases['layer1']

        predict = tf.matmul(
            hidden,
            self.weights['layer2']
            ) + self.biases['layer2']

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

    @define_scope
    def save_graph_ops(self):
        """
	    inputs:
		    graph_ops is a list of tensorflow graph operations
		    collection is a string
	    adds all the graph ops to the collection
	    """
        graph_ops = [
            self.x_in,
            self.y,
            self.optimizer,
            self.merged_summary_op,
            self.accuracy,
            self.model_pred,
            self.real_val
            ]

        for op in graph_ops:
	        tf.add_to_collection('my_ops', op)
	
        return

    def info(self):

        print("Model Name:        " + self.model_name)
        print("Embedding size:    " + str(self.embed_len))
        print("num_GRU_units:       " + str(self.num_GRU_units))
        print("num_layers:        " + str(self.num_layers))
        print("Input keep prob:   " + str(self.input_keep_prob))
        print("Output keep prob:  " + str(self.output_keep_prob))
