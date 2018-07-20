from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_processing import filter_activity, get_activity
import pickle


def process_key(key_np_array):
    keys_processed = np.empty([key_np_array.size, 14400])
    i = 0
    for patient in key_np_array:
        keys_processed[i] = filter_activity(get_activity(patient))
        i += 1
    return keys_processed


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')


def maxpool1d(x):
    return tf.nn.pool(x, [5], 'MAX', 'SAME', strides=[5])


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

# Initial Variables
batch_size = 50
epochs = 3

# Loading the outcome dataset and processing mesa-ids
with open('elements_file.pickle', 'rb') as handle:
    elements_file = pickle.load(handle)
with open('keys_unprocessed_file.pickle', 'rb') as handle:
    keys_file = pickle.load(handle)


# Splitting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(keys_file, elements_file, test_size=0.20, random_state=42)


# Input layer
x = tf.placeholder(tf.float32, [None, 14400], name='x')
y_ = tf.placeholder(tf.float32, [None, 1],  name='y_')
x_image = tf.reshape(x, shape=[-1, 14400, 1])

# Convolutional layer 1
W_conv1 = weight_variable([10, 1, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
h_pool1 = maxpool1d(h_conv1)
h_pool1 = tf.reshape(h_pool1, shape=[-1, 576, 5, 10])

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = maxpool2d(h_conv2)

# Fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 2300])
W_fc1 = weight_variable([2300, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output Layer
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')


# Evaluation functions
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_))
# Training algorithm
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Training steps
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    indices = np.arange(1, X_train.size)
    for epoch in range(epochs):
        indices = shuffle(indices)
        epoch_loss = 0
        i = 0
        while i < (len(X_train) - batch_size):
            batch = np.random.choice(indices, size=batch_size)
            batch_x = process_key(np.take(X_train, batch))
            batch_y = np.take(y_train, batch)
            batch_y = np.reshape(batch_y, (batch_y.size, 1))
            merge = tf.summary.merge_all()
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            epoch_loss += c
            i += batch_size
            print(str(i) + " Samples Processed")
        print('Epoch', epoch+1, 'completed out of', epochs, 'loss:', epoch_loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    print('Accuracy:', accuracy.eval({x: process_key(X_test), y_: y_test, keep_prob: 1.0}))
