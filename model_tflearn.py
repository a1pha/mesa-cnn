from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Establish random seed for partitioning train and test
np.random.seed(0)

# Splitting dataset into train and test
data = pd.read_csv('Data.csv')
data = data.drop(['mesaid'], axis=1)
data = data.drop(data.columns[0], axis=1)
perm = np.random.permutation(data.index)
m = len(data)
train_end = int(0.75 * m)
train = data.ix[perm[:train_end]]
test = data.ix[perm[train_end:]]

# Normalizing Data
X_train = (train.drop(['htn5c'], axis=1)).as_matrix
y_train = (train['htn5c']).as_matrix

X_test = (test.drop(['htn5c'], axis=1)).as_matrix
y_test = (test['htn5c']).as_matrix

# Model Parameters
n_classes = 1  # Corresponding to hypertension or not
batch_size = 150

y = tf.placeholder('float')

# Building convolutional network
network = input_data(shape=[None, 5*2880, 1], name='input')
network = conv_1d(network, 1, 10, activation='relu', regularizer="L2")
network = max_pool_1d(network, 5)
network = conv_1d(network, 1, 10, activation='relu', regularizer="L2")
network = max_pool_1d(network, 5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 1, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X_train}, {'target': y_train}, n_epoch=10,
           validation_set=({'input': X_test}, {'target': y_test}),
           snapshot_step=100, show_metric=True, run_id='mesa_1')