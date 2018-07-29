from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_processing import filter_activity, get_activity
import pickle

# Helper Method to Parse Mesa File Paths into Actigraphy Vectors
def process_input(key_np_array):
    keys_processed = np.empty([key_np_array.size, 14400])
    i = 0
    for patient in key_np_array:
        keys_processed[i] = filter_activity(get_activity(patient))
        i += 1
    return keys_processed

# Loading the data
with open('data_dict.pickle', 'rb') as read:
    data_dict = pickle.load(read)

labels = (np.array(list(data_dict.values()))).astype(int)
unprocessed_input = np.array(list(data_dict.keys()))

# Splitting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(unprocessed_input, labels, test_size=0.20, random_state=42)

# Converting labels to one-hot vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=25)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=25)

# Training Parameters
batch_size = 50
learning_rate = 0.01
epochs = 5
num_steps = X_train.size//batch_size

# Network Parameters
num_input = 14400
num_classes = 25
dropout = 0.75

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv1d(x, W, b, stride=1):
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool1d(x, k=5):
    return tf.nn.pool(x, [k], 'MAX', 'SAME', strides=[k])


# Store layers weight & bias
weights = {
    # 10x1 conv, 1 input, 10 outputs
    'wc1': tf.Variable(tf.random_normal([10, 1, 10])),
    #  conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 10, 20])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([2300, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([10])),
    'bc2': tf.Variable(tf.random_normal([20])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Define Model
def conv_net(x, weights, biases, dropout):
    # Reshaping inputs into proper shape
    x = tf.reshape(x, shape=[-1, 14400, 1])

    # First Convolutional Layer
    conv1 = tf.nn.relu(conv1d(x, weights['wc1'], biases['bc1'], stride=1))
    conv1 = maxpool1d(conv1, k=5)

    # Reshaping for 2D Convolutions
    conv1_reshaped = tf.reshape(conv1, shape=[-1, 5, 576, 10])

    # Second Convolutional Layer
    conv2 = tf.nn.relu(conv2d(conv1_reshaped, weights['wc2'], biases['bc2'], strides=1))
    conv2 = tf.squeeze(conv2, axis=1)
    conv2 = maxpool1d(conv2, k=5)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    indices = np.arange(1, X_train.size)
    for epoch in range(epochs+1):
        indices = shuffle(indices)
        epoch_loss = 0
        i = 0
        while i < (len(X_train) - batch_size):
            batch = np.random.choice(indices, size=batch_size)
            batch_x = process_input(np.take(X_train, batch))
            batch_y = np.take(y_train, batch, axis=0)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            i += batch_size
            print(str(i) + " Samples Processed")
        print("Epoch " + str(epoch+1) + " Completed")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: process_input(X_test),
                                        Y: y_test,
                                        keep_prob: 1.0}))