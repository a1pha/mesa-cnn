import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_processing import filter_activity, get_activity
import tensorflow as tf
import pickle

# Batch Size
batch_size = 50
n_classes = 1

# Loading the outcome dataset and processing mesa-ids
with open('elements_file.pickle', 'rb') as handle:
    elements_file = pickle.load(handle)
with open('keys_file.pickle', 'rb') as handle:
    keys_file = pickle.load(handle)


# Splitting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(keys_file, elements_file, test_size=0.20, random_state=42)


# Model Parameters
x = tf.placeholder('float', [None, 5*2880])
y = tf.placeholder('float')

def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool1d(x):
    return tf.nn.pool(x, [5], 'MAX', 'SAME', strides=[5])

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([10, 1, 10])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 10, 20])),
               'W_fc': tf.Variable(tf.random_normal([116000, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([10])),
              'b_conv2': tf.Variable(tf.random_normal([20])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 14400, 1])
    conv1 = conv1d(x, weights['W_conv1']) + biases['b_conv1']
    conv1 = maxpool1d(conv1)

    conv1 = tf.reshape(conv1, shape=[-1, 576, 5, 10])
    conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, shape=[-1, 116000])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']+biases['b_fc']))

    output = tf.matmul(fc, weights['out']+biases['out'])
    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size
                batch_x = np.array(X_train[start:end])
                batch_y = np.array(y_train[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))


train_neural_network(x)