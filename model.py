import tensorflow as tf
import pandas as pd
import numpy as np


# Dataset with outcomes
outcomes = "/Users/ajadhav0517/Box/mesa/mesa_nhlbi/Primary/Exam5/Data/mesae5_drepos_20151101.csv"
df = pd.read_csv(outcomes)
df = df[['mesaid', 'htn5c']].dropna()

df2 = pd.read_csv('Data.csv', header=None)
df2.rename(columns={0:'mesaid'}, inplace=True)


mergedDF = pd.merge(df, df2, on='mesaid', how='inner')
print(mergedDF.shape)
print(mergedDF.head())

n_classes = 2
batch_size = 128

x = tf.placeholder('float', [None, 5*2880])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv1d(x, W):
    return tf.nn.conv1d(x, W, strides=[10, 1, 1], padding='SAME')


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
              'b_fc': tf.Variable(tf.random_normal([500])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    conv1 = tf.nn.relu(conv1d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool1d(conv1)

    conv2 = tf.nn.relu(conv1d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool1d(conv2)

    fc = conv2
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)