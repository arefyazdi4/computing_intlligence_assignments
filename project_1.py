import numpy as np

if __name__ == '__main__':
    # NumPy Creating Arrays
    list_number_1 = [1, 2, 3, 4, 5]
    array_number_1 = np.array(list_number_1)
    print('numpy array_number_1', '\n', array_number_1, type(array_number_1))
    print('python list_number_1', '\n', list_number_1, type(list_number_1))
    print('numpy version', '\n', np.__version__)
    # Dimensions in Arrays
    d_0_array = np.array(42)
    print('0-D array', '\n', d_0_array)
    d_2_array_1 = np.array([[1, 2, 3], [4, 5, 6]])
    print('d_2_array_1', '\n', d_2_array_1)
    d_2_array_2 = np.array([1, 2, 3], ndmin=2)
    print('d_2_array_2', '\n', d_2_array_2)
    d_2_array_3 = np.array(np.mat('1 2 3; 4 5 6'))
    print('d_2_array_3', '\n', d_2_array_3, '\n')
    d_3_array = np.array([[[1, 5, 3], [4, 2, 6]], [[1, 2, 3], [7, 9, 8]]])
    print('d_3_array', '\n', d_3_array)
    # Check Number of Dimensions
    print('number of dimension:  ', d_0_array.ndim, array_number_1.ndim, d_2_array_1.ndim, d_3_array.ndim)

    # NumPy Array Indexing
    print('third element', array_number_1[2])
    print('Last element from 2nd dim', d_2_array_1[0, -1])

    # NumPy Array Slicing
    array_number_2 = np.array([1, 2, 3, 4, 5, 6, 7])
    print('slicing index 1 to 5: ', array_number_2[1:5])
    print('slicing 2d array\n', d_2_array_1[:, -1:-3:-1])

    # NumPy Data Types
    array_number_3 = np.array([1, 2], dtype='i4')
    # Checking the Data Type of an Array
    print('array 3 data type', array_number_3.dtype)
    # Creating Arrays With a Defined Data Type
    try:
        np.array(['a'], dtype='i')
    except ValueError as error:
        print('if data cant be covert:\n', error)
    # Converting Data Type on Existing Arrays
    array_number_4 = array_number_1.astype('S8')
    print('data type converting to string 8bit', array_number_4.dtype)
    array_number_5 = array_number_1.astype(float)
    print('data type int to float: ', array_number_5)

    # NumPy Array Copy vs View
    array_number_6_value = array_number_1.copy()
    array_number_7_reference = array_number_1.view()
    print('check if its original value or not\n', array_number_1.base, array_number_7_reference.base)

    # NumPy Array Shape
    print('array shape is its dimension values(k,i,j)', d_3_array.shape)
    # NumPy Array Reshaping
    d_1_array = arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    d_2_array_4_reshape = d_1_array.reshape(4, 3)
    print('reshape 1D array to 2D\n', d_2_array_4_reshape)
    d_3_array_2_reshape = d_1_array.reshape(2, 3, -1)  # -1 will let numpy to calculate last value it self
    print('reshape 1D to 3D\n', d_3_array_2_reshape)
    try:
        d_1_array.reshape(4, 4)
    except ValueError as error:
        print('reshape into any shape\n', error)
    #  reshape function returns a reference to original value
    print("d3 reshaped array reference", d_3_array_2_reshape.base)
    d_1_array_flatted = d_3_array_2_reshape.reshape(-1)
    print('3D to 1D flattening', d_1_array_flatted)
    print('rot90 2D (2,3) to (3,2) reverse clockwise\n', np.rot90(d_2_array_1))
    # print(np.flatiter(d_2_array_1))
    # print(np.ravel(d_1_array))
    print('flip (i,k)axis\n', np.flip(d_3_array, (2, 0)))
    # fliplr , flipud , second argument is axis (0:i,1:j,2:k,...)

    # NumPy Array Iterating
    print('iterate on 2D')
    for row in d_2_array_1:
        print(row)
    print('iterate using nditer function')
    d_3_array_4_simple = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    for data in np.nditer(d_3_array_4_simple):
        print(data, end=' ')
    print('iterate with changing data type and step 2')
    for data in np.nditer(d_3_array_4_simple[:, :, ::2], flags=['buffered'], op_dtypes=['S']):
        print(data, end=' ')
    print('iterate while enumerating')
    d_1_array_simple = np.array([1, 2, 3])
    for index, data in np.ndenumerate(d_1_array_simple):
        print(index, data)

    # NumPy Joining Array
    # in NumPy we join arrays by axis
    print('joining arrays content')
    d_2_array_5 = np.concatenate((d_2_array_1, d_2_array_3), axis=1)
    print('joint rows\n', d_2_array_5)
    d_2_array_6 = np.concatenate((d_2_array_1, d_2_array_2), axis=0)
    print('joint columns\n', d_2_array_6)
    print('done along a new axis')
    d_1_array_simple_2 = np.array([4, 5, 6])
    d_2_array_7 = np.stack((d_1_array_simple, d_1_array_simple_2), axis=1)
    print('axis 1\n', d_2_array_7)
    d_2_array_8 = np.stack((d_1_array_simple, d_1_array_simple_2), axis=0)
    print('axis 0\n', d_2_array_8)
    # hstack , vstack , hstack is other function to extend row,column and height
    d_2_array_9_split = np.array_split(d_1_array_simple, 2)

    # NumPy Splitting Array
    print('split array with 3 element to 2 array with 2 and 1 element\n', d_2_array_9_split)
    d_2_array_10 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    d_2_array_2i_3j = np.array_split(d_2_array_10, 3, axis=0)
    d_2_array_6i_1j = np.array_split(d_2_array_10, 3, axis=1)
    print('splitting same array in two different axis')
    print('axis i splitting rows\n', d_2_array_2i_3j)
    print('axis j splitting columns\n', d_2_array_6i_1j)
    # hsplit , vsplit , dsplit is other functions to split column , rows , height

    # NumPy Searching Arrays
    find_evens = np.where(array_number_1 % 2 == 0)
    print('find even num in array 1\n', find_evens)
    convert_array = np.where(array_number_1 < 3, array_number_1, array_number_1 * 10)  # where(condition,x,y)
    print('if condition is true yield x else yield y\n', convert_array)
    # Search Sorted
    index_to_insert = np.searchsorted(array_number_1, 2)  # find place to insert number 2 --> 1
    # The method starts the search from the left and returns the first index where
    # the number n is no longer larger than the next value
    print('binary search index to insert', index_to_insert)
    index_to_insert = np.searchsorted(array_number_1, 2, side='right')  # find place to insert number 2 --> 2
    print('binary search index from right side', index_to_insert)
    indexes_to_insert = np.searchsorted(array_number_1, [2, 3, 5])
    print('multiple values to search sorted', indexes_to_insert)

    # NumPy Sorting Arrays
    d_1_array_sorted = np.sort(array_number_1)
    # This method returns a copy of the array, leaving the original array unchanged.
    print('sorting array ascending', d_1_array_sorted)
    d_2_array_11 = np.array([[3, 2, 4], [5, 0, 1]])
    d_2_array_11_sorted = np.sort(d_2_array_11)
    print('2D array sorting\n', d_2_array_11_sorted)

    # NumPy Filter Array
    boolean_index_list = [True, False, False, True, False]
    filtered_array = array_number_1[boolean_index_list]
    print('filtered array with boolean index list', filtered_array)
    boolean_index_list = array_number_1 < 3
    filtered_array = array_number_1[boolean_index_list]
    print('filtered array with condition', filtered_array)
    boolean_index_list = list(map(lambda n: True if n % 2 == 0 else False, array_number_1))
    filtered_array = array_number_1[boolean_index_list]
    print('filtered array with map and lambda', filtered_array)
    print(d_3_array[0])
    print(d_3_array[0, :])
    print(np.sum(d_3_array, (1, 2)))
    print(np.argmax(d_3_array, axis=0))
    print(d_3_array.shape)