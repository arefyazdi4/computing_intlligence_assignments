import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
from sklearn.datasets import load_digits


class NearestNeighbor:
    def __init__(self):
        self.y_train_labels: np.ndarray = None  # label for each dataset
        self.x_train_datasets: np.ndarray = None  # list of datasets to compare as original data

    def train(self, train_datasets: np.ndarray, labels: np.ndarray):
        self.y_train_labels = labels
        self.x_train_datasets = train_datasets

    def predict(self, x_test_data: np.ndarray):
        tests_num = x_test_data.shape[0]
        trains_shape = self.x_train_datasets.shape
        trains_num = trains_shape[0]
        y_predicted_labels = np.zeros(tests_num, dtype=self.y_train_labels.dtype)

        for i in range(tests_num):
            distances_abs = np.zeros(trains_shape, dtype=self.x_train_datasets.dtype)
            for j in range(trains_num):  # broadcasting error
                distances_abs[j] = np.abs(self.x_train_datasets[j] - x_test_data[i])
            axis = axis_to_sum(trains_shape)  # needed axis to sum 2D->(1) 3D->(1,2)
            distances_sum = np.sum(distances_abs, axis=axis)
            min_distance_index = np.argmin(distances_sum)
            y_predicted_labels[i] = self.y_train_labels[min_distance_index]

        return y_predicted_labels


def axis_to_sum(shape):
    if len(shape) == 2:
        return 1
    else:
        return tuple(k + 1 for k in range(len(shape)-1))


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


if __name__ == '__main__':
    knn = NearestNeighbor()

    # load mnist with cvs -> kaggle
    # train dataset
    dataset_train_cvs = pd.read_csv('./mnistDataset/kaggle/train.csv')
    train_array: np.ndarray = dataset_train_cvs.iloc[:, 1:].values.astype('float32')  # all pixel values
    train_array = train_array.reshape(train_array.shape[0], 28, 28)
    train_label_array: np.ndarray = dataset_train_cvs.iloc[:, 0].values.astype('int32')
    # test dataset
    dataset_test_cvs = pd.read_csv('./mnistDataset/kaggle/test.csv')
    test_array: np.ndarray = dataset_test_cvs.iloc[:, :].values.astype('float32').reshape(-1, 28, 28)
    dataset_test_label_cvs = pd.read_csv('./mnistDataset/kaggle/mnist_submission.csv')
    test_label_array: np.ndarray = dataset_test_label_cvs.iloc[:, 1].values.astype('int32')
    # test the predict function
    knn.train(train_datasets=train_array, labels=train_label_array)
    predict_label_array = knn.predict(test_array[::500])
    print('\n\n kaggle dataset')
    print(predict_label_array)
    print('______Compare_______')
    print(test_label_array[::500])
    print(train_array.shape, test_array.shape)

    # load mnist with idx  -> document
    xtr = read_idx('./mnistDataset/mnist_duc/train-images.idx3-ubyte')
    ytr = read_idx('./mnistDataset/mnist_duc/train-labels.idx1-ubyte')
    xts = read_idx('./mnistDataset/mnist_duc/t10k-images.idx3-ubyte')
    yts = read_idx('./mnistDataset/mnist_duc/t10k-labels.idx1-ubyte')
    # test the predict function
    knn.train(xtr, ytr)
    y_predict = knn.predict(xts[::250])
    print('\n\n mnist dataset')
    print(y_predict)
    print('______Compare_______')
    print(yts[::250])
    print(xtr.shape, xts.shape)

    # load mnist with sklearn dataset
    digits = load_digits()
    # test the predict function
    knn.train(digits.data, digits.target)
    print('\n\nsklearn dataset')
    print(knn.predict(digits.data[::100]))
    print('______Compare_______')
    print(digits.target[::100])
    print(digits.data.shape)

