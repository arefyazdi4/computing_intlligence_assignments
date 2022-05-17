from tensorflow.keras.datasets import imdb
from tensorflow import keras
import numpy as np


def vectorized_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def decode_review(review: list[int]) -> str:
    # decode on of reviews
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in review])  # ? if value couldn't be find
    # 0 for padding , 1 for start , 2 for unknown
    return decoded_review


def encode_review(reviews: list[str]) -> list[list[int]]:
    word_index = imdb.get_word_index()
    encoded_reviews = list()
    for review in reviews:
        encoded_review: list[int] = [word_index.get(word, -1)+3 for word in review]
        encoded_review.insert(0, 1)
        encoded_reviews.append(encoded_review)
    return encoded_reviews


def grading_movie(predict_percent:float):
    grade = "unknown"
    if predict_percent >= 80:
        grade = "great movie"
    elif predict_percent >= 65:
        grade = 'it\'s good movie'
    elif predict_percent >= 40:
        grade = 'hard to tell how was it'
    else:
        grade = 'Couldn\'t be Worse than this'

    return grade


if __name__ == '__main__':
    # loading limited number of word
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # Encoding the integer sequences via one-hot encoding
    x_train = vectorized_sequences(train_data)
    x_test = vectorized_sequences(test_data)
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

    sample_review = ['the film were great it was just brilliant so much that i bought the film as soon as it was ' \
                     'released for and would recommend it to everyone to watch and the fly fishing was amazing really ' \
                     'cried at the end it was so sad and you know what they say if you cry at a film it must have been ' \
                     'good and this definitely was also to the two little boy that played the of norman and paul they ' \
                     'were just brilliant children are often left out of the list i think because the stars that play ' \
                     'them all grown up are such a big profile for the whole film but these children are amazing and ' \
                     'should be praised for what they have done you think the whole story was so lovely because it was ' \
                     'true and was someone life after all that was shared with ',
                     'this terrible movie i love cheesy horror movies and seen hundreds but this had got to be on of '
                     'the worst ever made the plot is paper thin and ridiculous the acting is an abomination the '
                     'script is completely laughable the best is the end showdown with the cop and how he worked out '
                     'who the killer is just so damn terribly written the clothes are sickening and funny in equal '
                     'the hair is big lots of boobs men wear those cut shirts that show off their sickening that men '
                     'actually wore them and the music is just trash that plays over and over again in almost every '
                     'scene there is trashy music boobs and taking away bodies and the gym still close for all joking '
                     'aside this is a truly bad film whose only charm is to look back on the disaster that was the 80 '
                     'and have a good old laugh at how bad everything was back then '
                     ]

    sample_reviews_encode = encode_review(sample_review)
    sample_review_vectorized = vectorized_sequences(sample_reviews_encode)

    predicts = model.predict(sample_review_vectorized)
    print(type(predicts))
    for index, predict in enumerate(predicts):
        print(f'review number{index}: ', grading_movie(predict))
