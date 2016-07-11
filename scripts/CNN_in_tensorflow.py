from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib import layers, learn

from sklearn.metrics import classification_report, confusion_matrix

path_to_project_data = '~/science/DSI/DSI-Capstone-Project/data/'

def scale_features(X):
    '''
    input: X (np array of any dimensions)
    cast as floats for division, scale between 0 and 1
    output: X (np array of same dimensions)
    '''
    X = X.astype("float32")
    X /= 255
    return X


def convert_targets(targets):
    '''
    input: targets (1D np array of strings)
    output: targets dummified category matrix
    note: targets are indexed as ['elliptical', 'merger', 'spiral']
    '''
    return pd.get_dummies(targets).values

'''
Best Net Architecture:
input: 60x60x3 RGB arrays scaled to [0,1]
SGD optimizer, learning rate eta=0.005 with decay 1e-6, Nesterov momentum mu=0.9
batch size 20
40 epochs with early stopping
dropout regularziation 0.5 in dense layers
loss function = categorical cross entropy
1) convolutional, 32 features, 5x5 filter, RELu, weights N(0, 0.01)
MaxPooling2D (2,2)
2) convolutional, 64 features, 5x5 filter, RELu, weights N(0, 0.01)
MaxPooling2D (2,2)
3) convolutional, 128 features, 3x3 filter, RELu, weights N(0, 0.01)
4) convolutional, 128 features, 3x3 filter, RELu, weights N(0, 0.1)
MaxPooling2D (2,2)
5) dense, 2048 features, maxout(2), weights N(0, 0.001)
6) dense, 2048 features, maxout(2), weights N(0, 0.001)
7) dense, 2048 features, maxout(2), weights N(0, 0.001)
8) dense, 2048 features, maxout(2), weights N(0, 0.001)
9) dense, 4, softmax
'''


tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

conv2d(*args, **kwargs)
Args:
  inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
  num_outputs: integer, the number of output filters.
  kernel_size: a list of length 2 `[kernel_height, kernel_width]` of
    of the filters. Can be an int if both values are the same.
  stride: a list of length 2 `[stride_height, stride_width]`.
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
  padding: one of `VALID` or `SAME`.
  activation_fn: activation function.
  normalizer_fn: normalization function to use instead of `biases`. If
    `normalize_fn` is provided then `biases_initializer` and
    `biases_regularizer` are ignored and `biases` are not created nor added.
  normalizer_params: normalization function parameters.
  weights_initializer: An initializer for the weights.
  weights_regularizer: Optional regularizer for the weights.
  biases_initializer: An initializer for the biases. If None skip biases.
  biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
  variables_collections: optional list of collections for all the variables or
    a dictionay containing a different list of collection per variable.
  outputs_collections: collection to add the outputs.
  trainable: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
  scope: Optional scope for `variable_op_scope`.



def inference(images):
    '''
    input:
    output:
    '''
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def train_model(n_batches, batch_size, dropout = 0.5):
    '''
    for mnist example, model took a really long time for 20000 (suggested
    number) and seemed to have fit well by step 1200
    input: n_batches (int), batch_size (int)
    output: None (prints accuracy to screen)
    '''
    # Adam Optimizer with step size of 0.0001
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    for i in range(n_batches):
        # figure out how to do this with regular syntax (not mnist built in)
        batch = mnist.train.next_batch(batch_size)
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, \
                            len(trX), batch_size))

        # every 100 turns = 1 full batch (5000 images)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], \
            y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1 - dropout})


        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          keep_prob: 1 - dropout})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

if __name__ == '__main__':
    # Launch the session
    with tf.Session() as sess:
