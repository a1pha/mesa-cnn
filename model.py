import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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

x = tf.placeholder('float', [None, 5*2880])
y = tf.placeholder('float')

# keep_rate = 0.8
# keep_prob = tf.placeholder(tf.float32)
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=10, padding='SAME')


def maxpool1d(x):
    #                        size of window         movement of window
    return tf.layers.max_pooling1d(x, pool_size=5, strides=5,padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([10, 1, 1])),
               'W_conv2': tf.Variable(tf.random_normal([10, 1, 1])),
               'W_fc': tf.Variable(tf.random_normal([576, 500])),
               'out': tf.Variable(tf.random_normal([500, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([1])),
              'b_conv2': tf.Variable(tf.random_normal([1])),
              'b_fc': tf.Variable(tf.random_normal([250])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    conv1 = tf.nn.relu(conv1d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool1d(conv1)

    conv2 = tf.nn.relu(conv1d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool1d(conv2)

    fc = conv2
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(X_train)/batch_size)):
                epoch_x, epoch_y = next_batch(batch_size, X_train, y_train)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))


train_neural_network(x)