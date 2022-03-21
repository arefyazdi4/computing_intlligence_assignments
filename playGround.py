class Data:
    def __init__(self):
        self.num = 1
        self.list = [1, 2]

    def __str__(self):
        return self.num.__str__() + '  ' + self.list.__str__()


def check_is_pointer(var):
    var.num = var.num + 1
    var.list.append(3)


def idx_to_array():
    import os
    import codecs
    import numpy as np
    # PROVIDE YOUR DIRECTORY WITH THE EXTRACTED FILES HERE
    datapath = './mnistDataset/mnist_loader/idx/'

    files = os.listdir(datapath)

    def get_int(b):  # CONVERTS 4 BYTES TO A INT
        return int(codecs.encode(b, 'hex'), 16)

    data_dict = {}
    for file in files:
        if file.endswith('ubyte'):  # FOR ALL 'ubyte' FILES
            print('Reading ', file)
            with open(datapath + file, 'rb') as f:
                data = f.read()
                type = get_int(data[:4])  # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
                length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
                if type == 2051:
                    category = 'images'
                    num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                    num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)  # READ THE PIXEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length, num_rows,
                                            num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]
                elif type == 2049:
                    category = 'labels'
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)  # READ THE LABEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]
                if length == 10000:
                    set = 'test'
                elif length == 60000:
                    set = 'train'
                data_dict[set + '_' + category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY
    return data_dict


if __name__ == '__main__':
    # var_original = Data()
    # check_is_pointer(var_original)
    # print(var_original)

    #  primary data like str,int,float ,... only pass by value
    #  but non primary like list or class objects have been passed by reference

    # convert cvs file to nd array
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # test_cvs = pd.read_csv('./mnistDataset/kaggle/mnist_submission.csv')
    # test: np.ndarray = test_cvs.iloc[:,1].values.astype('int32')  # all pixel values
    # test = test.reshape(test.shape[0], 28, 28)
    #
    # i = 10
    # # plt.imshow(test[i], cmap=plt.get_cmap('gray'))
    # # plt.waitforbuttonpress()
    # print(test.shape)

    #
    # file = open(r'./mnistDataset/kaggle/sample_submission.csv')
    # numpy_array = np.loadtxt(file, delimiter=",")
    # print(numpy_array)

    arra1 = np.array([1,2,3,4,5])
    arra2 = np.array([2,2,2,2,2])
    print(arra1[np.argmin(np.abs(arra2-arra1))])
    arra3 = np.array([[1,1,0],[2,0,3],[0,0,4],[0,0,0]])
    print(np.sum(arra3, axis=0))