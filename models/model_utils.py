"""Helper functions for constructing models"""

from os import makedirs, path
import tensorflow as tf
import numpy as np

# Data functions
def load_model_params(filepath):
    """Loads model values from path"""
    return np.load(filepath)

def save_model_params(params, filepath):
    """Saves model values to path"""
    filepath_dir = path.dirname(filepath)
    if not path.exists(filepath_dir):
        makedirs(filepath_dir)

    np.save(filepath, params)

# Variable functions
def weights_var(value=None, shape=None):
    """Weights tensor"""
    if value != None:
        return tf.Variable(value)
    if shape != None:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def biases_var(value=None, shape=None):
    """Biases tensor"""
    if value != None:
        return tf.Variable(value)
    if shape != None:
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

# Layer functions
def conv2D(x, W, name=None):
    """2D Convolution"""
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    """2x2 spacial max pool"""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],
                          padding='SAME', name=name)

def avg_pool():
    pass

def max_pool():
    pass

def relu(x, b, name=None):
    """RELU activation"""
    return tf.nn.relu(x + b, name=name)

def batch_norm():
    pass

def fc_layer(x, W, b, relu_activation=True, name=None):
    """Fully connected layer with optional RELU activation"""
    if relu_activation:
        return relu(tf.matmul(x, W), b, name=name)
    else:
        return tf.matmul(x, W, name=name)
