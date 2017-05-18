"""Classification training"""

import os, sys
sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../util/')

import time
import json
import importlib
import argparse

import tensorflow as tf
import numpy as np

from input_data import Data

def optimization(learning_rate, loss):
    """Defines the optimization operation"""
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def cross_entropy(logits, labels):
    """Defines loss function"""
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)

def compute_accuracy(logits, labels):
    """Computes classification accuracy"""
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def main():
    parser = argparse.ArgumentParser("Classification training.")
    parser.add_argument('model_config_file')
    parser.add_argument('num_iters', type=int)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('--model_snapshot_dir', default='../snapshots')

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
    train_data = Data(config['data']['train'])
    valid_data = Data(config['data']['valid'])
    train_data.shuffle_data()

    # Snapshot setup
    snapshot_filename = model_name + '.npy'
    snapshot_path = os.path.join(args.model_snapshot_dir,
                                 model_name,
                                 snapshot_filename)

    # Importing and creating model
    model_module = importlib.import_module(model_name)
    model = model_module.get_model(class_count,
                                   input_size_w=input_size_w,
                                   input_size_h=input_size_h,
                                   is_training=True)

    # Define loss, training, and validation functions
    loss = cross_entropy(model.logits, model.labels)
    train_step = optimization(args.learning_rate, loss)
    accuracy = compute_accuracy(model.logits, model.labels)

    # Session and variable initialization
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Training code here
    print('Begin Training')

    elapsed_time = 0.0
    initial_time = time.time()
    for i in range(args.num_iters):
        i += 1

        train_images, train_labels = train_data.next_batch(batch_size)
        train_step.run(feed_dict={model.x: train_images,
                                  model.labels: train_labels,
                                  model.dropout_prob: dropout})

        if i % 100 == 0:
            step_time = time.time() - initial_time
            print(i, 'iterations: took {0:.2f}s'.format(step_time))
            elapsed_time += step_time
            initial_time = time.time()

        if i % 500 == 0:
            valid_images, valid_labels = valid_data.next_batch(batch_size)
            acc = accuracy.eval(feed_dict={model.x: valid_images,
                                           model.labels: valid_labels,
                                           model.dropout_prob: 1.0})
            print('Training accuracy: {0:.2f}%'.format(acc * 100))
            print('Saving snapshot to', snapshot_path)
            model.save_params(sess, snapshot_path)

    print('Done')
    print('Took {0:.2f}s'.format(elapsed_time))

if __name__ == "__main__":
    main()
