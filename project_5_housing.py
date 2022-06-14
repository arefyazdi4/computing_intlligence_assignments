from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Model definition
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
        # no activation (it will be a linear layer)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # A metric is a function that is used to judge the performance of your model.
    # Metric functions are similar to loss functions, except that
    # the results from evaluating a metric are not used when training the model.
    return model


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print(train_data.shape)
    print(test_data.shape)
    # Normalizing the data
    # you subtract the mean of the feature and divide by the standard deviation
    # so that the feature is centered around 0 and has a unit standard deviation
    # quantities used for normalizing the test data are computed using the training data.
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data -= mean
    train_data /= std
    test_data -= mean
    test_data /= std
    # K-fold cross-validation
    # validation scores might change a lot depending on which data points
    # you chose to use for validation and which you chose for training
    # This would prevent you from reliably evaluating your model.
    # The validation score for the model used is then the average of the validation scores obtained
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print('processing fold #%d' % i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)
    # Building the history of successive mean K-fold validation scores
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    # Plotting validation scores
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    # Plotting smoothed validation scores, excluding the first 10 data points
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
    # Training the final model
    model = build_model()
    model.fit(train_data, train_targets,
              epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print(test_mae_score)

    predictions = model.predict(test_data)
    print(predictions[0])