"""AlexNet cnn model"""

import numpy as np
import tensorflow as tf

from model_utils import *
from model import Model

class AlexNet(Model):

    def __init__(self, class_count, input_size_w=28,
                 input_size_h=None, params_path=None,
                 is_training=False):
        """Constructor.
        Instantiates initial params"""
        super(AlexNet, self).__init__(params_path)
        self.input_size = [input_size_w, input_size_h or input_size_w]
        self.class_count = class_count
        self.dropout_prob = tf.placeholder(tf.float32)

        self.create(self.params, is_training=is_training)

    def save_params(self, sess, filepath):
        """Evaluates model params and saves them to file"""
        values = []
        for i in range(self.params.shape[0]):
            row_values = []
            for j in range(self.params.shape[1]):
                value = self.params[i][j].eval(sess)
                row_values.append(value)
            values.append(row_values)

        super(AlexNet, self).save_params(values, filepath)

    def create(self, params, is_training=False):
        # Load model parameters if they exist by calling parent class
        super(AlexNet, self).create(params)

        # Input layer
        self.x = tf.placeholder(tf.float32,
                           [None,self.input_size[0] * self.input_size[1]])
        x_img = tf.reshape(self.x, [-1,self.input_size[0],self.input_size[1],1])
        print('Input:', x_img.get_shape())

        # Conv2D layer 1
        conv1_W = weights_var(value=self.weights[0], shape=[5,5,1,32])
        conv1_b = biases_var(value=self.biases[0], shape=[32])
        conv1 = relu(conv2D(x_img, conv1_W), conv1_b, name='conv1')
        pool1 = max_pool_2x2(conv1, name='pool1')
        print('Conv1:', pool1.get_shape())

        # Conv2D layer 2
        conv2_W = weights_var(value=self.weights[1], shape=[5,5,32,64])
        conv2_b = biases_var(value=self.biases[1], shape=[64])
        conv2 = relu(conv2D(pool1, conv2_W), conv2_b, name='conv2')
        pool2 = max_pool_2x2(conv2, name='pool2')
        print('Conv2:', pool2.get_shape())

        # Conv2D layers 3-5
        conv3_W = weights_var(value=self.weights[2], shape=[5,5,64,128])
        conv3_b = biases_var(value=self.biases[2], shape=[128])
        conv3 = relu(conv2D(pool2, conv3_W), conv3_b, name='conv3')
        conv4_W = weights_var(value=self.weights[3], shape=[5,5,128,256])
        conv4_b = biases_var(value=self.biases[3], shape=[256])
        conv4 = relu(conv2D(conv3, conv4_W), conv4_b, name='conv4')
        conv5_W = weights_var(value=self.weights[4], shape=[5,5,256,512])
        conv5_b = biases_var(value=self.biases[4], shape=[512])
        conv5 = relu(conv2D(conv4, conv5_W), conv5_b, name='conv5')
        pool5 = max_pool_2x2(conv5, name='pool5')
        print('Conv5:', pool5.get_shape())

        # FC layer 6
        w = int(round(self.input_size[0] / 2 / 2 / 2))
        h = int(round(self.input_size[1] / 2 / 2 / 2))
        fc6_W = weights_var(value=self.weights[5], shape=[w*h*512,1024])
        fc6_b = biases_var(value=self.biases[5], shape=[1024])
        pool5_flat = tf.reshape(pool5, [-1,w*h*512])
        fc6 = fc_layer(pool5_flat, fc6_W, fc6_b, name='fc6')
        print('FC6:', fc6.get_shape())

        fc6_dropout = tf.nn.dropout(fc6, self.dropout_prob)

        # FC layer 7
        fc7_W = weights_var(value=self.weights[6], shape=[1024,512])
        fc7_b = biases_var(value=self.biases[6], shape=[512])
        fc7 = fc_layer(fc6_dropout, fc7_W, fc7_b, name='fc7')
        print('FC7:', fc7.get_shape())

        fc7_dropout = tf.nn.dropout(fc7, self.dropout_prob)

        # Logits layer
        logits_W = weights_var(value=self.weights[7], shape=[512,self.class_count])
        logits_b = biases_var(value=self.biases[7], shape=[self.class_count])
        self.logits = tf.matmul(fc7_dropout, logits_W, name='logits') + logits_b
        print('Logits:', self.logits.get_shape())

        # True labels
        self.labels = tf.placeholder(tf.float32, [None,self.class_count])

        # Loads/instantiates weights and biases
        if is_training:
            self.weights = np.array([conv1_W, conv2_W,
                                     conv3_W, conv4_W,
                                     conv5_W, fc6_W,
                                     fc7_W, logits_W])
            self.biases = np.array([conv1_b, conv2_b,
                                    conv3_b, conv4_b,
                                    conv5_b, fc6_b,
                                    fc7_b, logits_b])
            self.params = np.array([self.weights, self.biases])

def get_model(class_count, input_size_w=28, input_size_h=None,
              params_path=None, is_training=False):
    return AlexNet(class_count, input_size_w, input_size_h,
                   params_path, is_training=is_training)

def main():
    my_model = AlexNet(10)

if __name__ == "__main__":
    main()
