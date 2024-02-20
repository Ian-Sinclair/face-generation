from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np

class GANLosses(losses.Loss):    
    def __init__(self):
        pass

    @staticmethod
    def wasserstein(y_true, y_pred):
        return -K.mean(y_true * y_pred)
    
    @staticmethod
    #@tf.function()
    def gradient_penalty_loss(gradients):
        # compute the euclidean norm by squaring ...
        gradients_sqr = tf.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr,
                                        axis=list(range(1, len(gradients_sqr.shape))))
        #   ... and sqrt
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        
        return tf.reduce_mean(gradient_penalty)