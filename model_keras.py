import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Reshape, Dropout
from keras.models import Sequential
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from data_processing import filter_activity, get_activity

# Helper Method to Parse Mesa File Paths into Actigraphy Vectors
def process_input(key_np_array):
    keys_processed = np.empty([key_np_array.size, 14400])
    i = 0
    for patient in key_np_array:
        keys_processed[i] = filter_activity(get_activity(patient))
        i += 1
    return keys_processed

# Training Parameters
batch_size = 50
num_classes = 25
epochs = 1

# input image dimensions
img_x = 14400

# Loading the dataset
with open('data_dict.pickle', 'rb') as read:
    data_dict = pickle.load(read)

labels = (np.array(list(data_dict.values()))).astype(int)
unprocessed_input = np.array(list(data_dict.keys()))

# Splitting the data into training and test
X_train, X_test, y_train, y_test = train_test_split(unprocessed_input, labels, test_size=0.33)
# Reshaping Data
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
y_train = keras.utils.to_categorical(y_train, num_classes=25)
y_test = keras.utils.to_categorical(y_test, num_classes=25)


model = Sequential()
model.add(Conv1D(10, kernel_size=10, strides=1,
                 activation='sigmoid',
                 input_shape=input_shape, padding='same'))
model.add(MaxPooling1D(pool_size=5, strides=5))
model.add(Reshape((5, 576, 10)))
model.add(Conv2D(20, (5, 5), activation='sigmoid', padding='valid'))
model.add(MaxPooling2D(pool_size=(1, 5)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
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
          shuffle=True)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, (epochs+1)), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

print(model.summary())