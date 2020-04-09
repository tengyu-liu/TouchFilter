import numpy as np
import tensorflow as tf
import tf_util

class HandPriorNN:
    def __init__(self, l2_reg):
        self.l2_reg = l2_reg
        pass
    
    def __call__(self, z):
        with tf.variable_scope('HandPriorNN', reuse=tf.AUTO_REUSE):
            h1 = tf_util.fully_connected(z, 64, activation_fn=tf.nn.relu, weight_decay=self.l2_reg, scope='hand_prior_1')
            h2 = tf_util.fully_connected(h1, 64, activation_fn=tf.nn.relu, weight_decay=self.l2_reg, scope='hand_prior_2')
            prior = tf_util.fully_connected(h2, 1, activation_fn=None, weight_decay=self.l2_reg, scope='hand_prior_3')
            return prior
