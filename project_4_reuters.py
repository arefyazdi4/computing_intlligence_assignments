from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow.python.keras.utils
import matplotlib.pyplot as plt


def vectorized_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for indexes in sequence:
            if -1 < indexes < 10000:
                results[i, indexes] = 1.
    return results


# Encoding the labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


def decoded_newswire(news):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(word_index - 3, '?') for word_index in news])
    return decoded_newswire


def encode_news(news: list[str]) -> list[list[int]]:
    word_index = reuters.get_word_index()
    encoded_news = list()
    for new in news:
        encoded_new: list[int] = [word_index.get(word, -1) + 3 for word in new]
        encoded_new.insert(0, 1)
        encoded_news.append(encoded_new)
    return encoded_news


if __name__ == '__main__':
    # Classifying news wires: a multiclass classification example

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
    # Training the model
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    # Plotting the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Plotting the training and validation accuracy
    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # Retraining a model from scratch
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(46, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(partial_x_train,
              partial_y_train,
              epochs=9,
              batch_size=512,
              validation_data=(x_val, y_val))
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)

    sample_news = [
        'the european community ec agreed tough new rules to cut diesel from trucks and buses in an attempt to reduce air pollution threatening vast stretches of the region diplomats said ec environment ministers meeting here agreed member states would have to reduce by 20 pct over the next few years the emission of nitrogen oxide widely seen as the main source of acid rain endangering and lakes the reduction would be for heavy vehicles with tougher standards imposed for new models from april 1988 and for all new vehicles from october 1990 the ec executive commission says the emission level of nitrogen oxide was expected to drop to 2 4 mln tonnes a year from three mln tonnes within the 12 nation community if all heavy vehicles applied to the new standards there are an estimated nine and buses in use in the ec according to commission figures the ministers also gave west germany a go ahead to move towards a ban on the sale of lead regular petrol after bonn requested permission to do so to encourage the use of low pollution cars diplomats said west germany will still need ministers final approval for such a plan diplomats said this was expected when ec environment ministers meet next on may 21 but the ministers added that the go ahead for west germany did not mean there would automatically follow a community wide ban on the sale of regular leaded petrol bonn intends to keep leaded premium petrol pumps diplomats said they added that of the 97 mln cars in the ec only 20 mln now ran on regular leaded petrol and these would risk no damage if they switched over to premium leaded petrol under ec law ministers have to give member states special permission if they wish to be exempt from community competition laws this would be the case if west germany were to implement a ban on the sale of leaded regular petrol reuter',
        'the first national bank of boston the main banking unit of bank of boston said it is raising its prime lending rate to 7 75 pct from 7 50 pct effective immediately',
        'drought has resulted in a reduction in china estimated wheat crop this year to 87 0 mln tonnes 2 0 mln below last year harvest the u s agriculture department officer in peking said in a field report the report dated march 25 said imports in the 1987 88 season are projected at 8 0 mln tonnes 1 0 mln tonnes above the the current season estimate imports from the united states are estimated at 1 5 mln tonnes compared to only 150 000 tonnes estimated for the 1986 87 year it said after to major wheat producing areas and obtaining more information on the planted area the total planted area was estimated down 290 000 hectares due to the dry fall it said the report said to compensate for the below normal precipitation irrigation has increased as has the use of fertilizer while there are where irrigation is not possible most of the wheat crop has access to some water and therefore has emerged from dormancy and is doing well the report said it said scattered rain in many parts of china in the past 10 days has improved the situation but information on damage in anhui is',
    ]

    sample_news_encode = encode_news(sample_news)
    sample_news_vectorized = vectorized_sequences(sample_news_encode)
    predicts = model.predict(sample_news_vectorized)

    for index, predict in enumerate(predicts):
        predicted_category = np.argmax(predict)
        print(f'predicted_category number#{index}', predicted_category)
