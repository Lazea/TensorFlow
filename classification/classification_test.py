"""Classification testing"""

import os, sys
sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../util/')
import cv2
import time
import json
import importlib
import argparse

import tensorflow as tf
import numpy as np

from input_data import Data

def compute_accuracy(logits, labels):
    """Computes classification accuracy"""
    correct_prediction = correct_prediction(logits, labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def correct_prediction(logits, labels):
    """Computes the number of correct predictions"""
    return tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))

def main():
    parser = argparse.ArgumentParser("Classification testing.")
    parser.add_argument('model_config_file')
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--model_snapshot_dir', type=str, default=None)

    args = parser.parse_args()

    # Loads config file
    config = json.load(open(args.model_config_file, 'r'))
    model_name = config['model']
    input_size_w = config['input_size']['x']
    input_size_h = config['input_size']['y']
    input_channels = config['input_size']['channel']
    class_count = config['class_count']
    batch_size = config['batch_size']
    dropout = config['dropout']

    # Loading data
    input_path = config['data']['test']
    if args.images_dir != None:
        input_path = args.images_dir

    test_data = Data(input_path)
    image_count = test_data.count

    print('Found {} images'.format(image_count))

    # Load model
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(class_count,
                                   input_size_w=input_size_w,
                                   input_size_h=input_size_h,
                                   params_path=args.model_snapshot_dir)
    logits = model.logits
    b = model.biases


    correct = correct_prediction(model.logits, model.labels)

    # Session and variable initialization
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Compute accuracy of test data
    score = 0
    for i in range(image_count):
        test_images, test_labels = test_data.next_batch(1)
        pred = logits.eval(feed_dict={model.x: test_images,
                                      model.labels: test_labels,
                                      model.dropout_prob: 1.0})

        #print('T:', np.argmax(test_labels[0]), 'P:', np.argmax(pred))
        corr = correct.eval(feed_dict={model.x: test_images,
                                       model.labels: test_labels,
                                       model.dropout_prob: 1.0})
        if corr:
            score += 1

    print('Test accuracy: {}%'.format(round(score / image_count, 2)))


if __name__ == "__main__":
    main()
