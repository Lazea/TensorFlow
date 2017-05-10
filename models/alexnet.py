"""AlexNet cnn model"""

import numpy as np
import tensorflow as tf

from model_utils import *
from model import Model

class AlexNet(Model):

    def __init__(self, class_count, input_size_w=28, input_size_h=None, params_path=None):
        """Constructor.
        Instantiates initial params"""
        super(AlexNet, self).__init__(params_path)
        self.input_size = [input_size_w, input_size_h or input_size_w]
        self.class_count = class_count

        self.create(self.params)

    def create(self, params):
        # Input layer
        x = tf.placeholder(tf.float32,
                           [None,self.input_size[0] * self.input_size[1]])
        x_img = tf.reshape(x, [-1,self.input_size[0],self.input_size[1],1])
        print('Input:', x_img.get_shape())

        # Conv2D layer 1
        conv1_W = weights_var([5,5,1,32])
        conv1_b = biases_var([32])
        conv1 = relu(conv2D(x_img, conv1_W), conv1_b, name='conv1')
        pool1 = max_pool_2x2(conv1, name='pool1')
        print('Conv1:', pool1.get_shape())

        # Conv2D layer 2
        conv2_W = weights_var([5,5,32,64])
        conv2_b = biases_var([64])
        conv2 = relu(conv2D(pool1, conv2_W), conv2_b, name='conv2')
        pool2 = max_pool_2x2(conv2, name='pool2')
        print('Conv2:', pool2.get_shape())

        # Conv2D layers 3-5
        conv3_W = weights_var([5,5,64,128])
        conv3_b = biases_var([128])
        conv3 = relu(conv2D(pool2, conv3_W), conv3_b, name='conv3')
        conv4_W = weights_var([5,5,128,256])
        conv4_b = biases_var([256])
        conv4 = relu(conv2D(conv3, conv4_W), conv4_b, name='conv4')
        conv5_W = weights_var([5,5,256,512])
        conv5_b = biases_var([512])
        conv5 = relu(conv2D(conv4, conv5_W), conv5_b, name='conv5')
        pool5 = max_pool_2x2(conv5, name='pool5')
        print('Conv5:', pool5.get_shape())

        # FC layer 6
        fc6_W = weights_var([4*4*512,1024])
        fc6_b = biases_var([1024])
        pool5_flat = tf.reshape(pool5, [-1,4*4*512])
        fc6 = fc_layer(pool5_flat, fc6_W, fc6_b, name='fc6')
        print('FC6:', fc6.get_shape())

        dropout = tf.placeholder(tf.float32)
        fc6_dropout = tf.nn.dropout(fc6, dropout)

        # FC layer 7
        fc7_W = weights_var([1024,512])
        fc7_b = biases_var([512])
        fc7 = fc_layer(fc6_dropout, fc7_W, fc7_b, name='fc7')
        print('FC7:', fc7.get_shape())

        fc7_dropout = tf.nn.dropout(fc7, dropout)

        # Logits layer
        logits_W = weights_var([512,self.class_count])
        logits_b = biases_var([self.class_count])
        y = tf.matmul(fc7_dropout, logits_W, name='logits') + logits_b
        print('Logits:', y.get_shape())

        # True labels
        y_ = tf.placeholder(tf.float32, [None,self.class_count])

        # Loads/instantiates weights and biases
        if self.params != None:
            self.weights, self.biases = self.get_weights_and_biases(self.params)
        else:
            self.weights = np.array([conv1_W, conv2_W,
                                     conv3_W, conv4_W,
                                     conv5_W, fc6_W,
                                     fc7_W, logits_W])
            self.biases = np.array([conv1_b, conv2_b,
                                    conv3_b, conv4_b,
                                    conv5_b, fc6_b,
                                    fc7_b, logits_b])

def get_model(class_count):
    return AlexNet(class_count)

def main():
    my_model = AlexNet(10)

if __name__ == "__main__":
    main()
