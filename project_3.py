from tensorflow.keras.datasets import imdb
from tensorflow import keras
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


if __name__ == '__main__':
    # loading limited number of word
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(train_data[0])
    # decode on of reviews
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])  # ? if value couldn't be find
    print(decoded_review)
    # Encoding the integer sequences via one-hot encoding
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    print(x_train[0])
    # vectorized your labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    # Model definition
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # Setting aside a validation set
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    # Compiling the model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(results)