from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Reshape, Dropout
from keras.models import Sequential
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import pickle
from keras.callbacks import TensorBoard

batch_size = 100
num_classes = 2
epochs = 1

# input image dimensions
img_x = 14400

# Loading the outcome dataset and processing mesa-ids
with open('elements_file.pickle', 'rb') as handle:
    elements_file = pickle.load(handle)
with open('keys_file.pickle', 'rb') as handle:
    keys_file = pickle.load(handle)

# Splitting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(keys_file, elements_file, test_size=0.20, random_state=42)
# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
X_train = X_train.reshape(X_train.shape[0], img_x, 1)
X_test = X_test.reshape(X_test.shape[0], img_x, 1)
input_shape = (img_x, 1)

# convert the data to the right type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv1D(10, kernel_size=10, strides=1,
                 activation='sigmoid',
                 input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(pool_size=5, strides=5))
model.add(Reshape((576, 5, 10)))
model.add(Conv2D(20, (5, 5), activation='sigmoid', padding='same'))
model.add(MaxPooling2D(pool_size=(5, 1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=1e-4, nesterov=False),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[history],
          shuffle=True,)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, (epochs+1)), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
