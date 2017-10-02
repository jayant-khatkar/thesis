"""
GRU_net with fully connected Model

this model loads the weights from a CNN2 model
and feed the output of the pretrained convolutions into 
GRU_net

Author: Jayant Khatkar
"""

import tensorflow as tf
import numpy as np
import os
from scope_decorator import *
from GRU_net import *

class GRU_CNN:

    def __init__(self, save_path, hyperPs):
        """
        restores model at string save_path, and feeds the output into GRU_net
        """
		
		sess = tf.Session()
		
		graph_ops = restorer(save_path, sess, 'my_ops')
		placeholders = [graph_ops[u'x_in:0'], graph_ops[u'target:0']]
		
        self.x_in = placeholders[0]
        self.y    = placeholders[1]
		
		GRU_in = graph_ops[u'prediction/GRU_net_input:0']

        self.n_classes  = self.y.get_shape()[1]
        self.n_words    = GRU_in.get_shape()[1]
        self.embed_len  = GRU_in.get_shape()[2]
        self.model_name = "GRU_CNN"
		
		RNN = GRU_net(
	        GRU_in, 
			self.y,
            self.embed_len,
			self.n_words,
			self.n_classes,
			hyperPs)

        self.prediction
        self.optimize
        self.measure_performance
        self.tensorboard_summary
        self.save_graph_ops
		
		return sess

    @define_scope
    def prediction(self):
        return RNN.prediction


    @define_scope
    def optimize(self):
        self.optimizer = RNN.optimizer
        return self.optimizer


    @define_scope
    def measure_performance(self):

        self.model_pred = RNN.model_pred
        self.real_val   = RNN.real_val
        self.accuracy   = RNN.accuracy

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
