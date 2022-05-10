from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow.python.keras.utils


def vectorized_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Encoding the labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    x_train = vectorized_sequences(train_data)
    x_test = vectorized_sequences(test_data)
    # Encoding the labels
    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)
    # Model definition
    # num of layers are collected from hyper parameters by test and gathering info from data validation
    # reason behind using relu rather than sigmoid is dense of layer and preventing errors to pass layers to layers
    # soft max other than sigmoid cuzed to result be more reliable
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(46, activation='softmax')
    ])

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

