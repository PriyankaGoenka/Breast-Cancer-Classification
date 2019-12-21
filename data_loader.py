import numpy as np
def load_data(file_path, test_size=0.5, random_seed=0):
    """
    Function to load the data and create train an test sets.
    Input:
        - input_file: path to input file.
        - test_size: a floating point number between 0.0 and 1.0, which represents the proportion of 
        the input data to be used as the test set.
        - random_seed:  an integer seed to be used by the random number generator. 
    Output:
        - X_train: a n*9 predictor matrix for training the classifier. n is training size. 
        - Y_train: a n*1 response matrix for training.
        - X_test: a m*9 predictor matrix for testing the classifier. m is test size size. 
        - Y_test: a m*1 response matrix for testing.
    """
    np.random.seed(random_seed)
    OriginalData = np.loadtxt(file_path, delimiter=',')
    OriginalData = OriginalData[:, 1:]
    OriginalData = np.take(OriginalData, np.random.permutation(OriginalData.shape[0]), axis=0,
                           out=OriginalData)  # Shuffle rows
    nrows, ncols = np.shape(OriginalData)
    a = np.round(nrows * test_size)
    TestData = OriginalData[0:int(a), :]
    TrainData = OriginalData[int(a):nrows, :]

    X_train = TrainData[:, 0:ncols - 1]
    X_test = TestData[:, 0:ncols - 1]
    Y_train = TrainData[:, ncols - 1:ncols]
    Y_test = TestData[:, ncols - 1: ncols]

    for i in range(np.shape(X_train)[0]):
        if int(Y_train[i, 0]) == 2:
            Y_train[i, 0] = 0
        else:
            Y_train[i, 0] = 1

    for i in range(np.shape(X_test)[0]):
        if int(Y_test[i, 0]) == 2:
            Y_test[i, 0] = 0
        else:
            Y_test[i, 0] = 1

    
    ################################################################################################
    # TODO: (1) load the input data given the file_path.                                           #
    # (2) Extract the  predictor  and outcome variable. The outcome should be converted to 0 and 1 #
    # (3) Create train and test set.                                                               #
    ################################################################################################
    # START OF YOUR CODE

    return X_train, X_test, Y_train, Y_test

#print(load_data('BCW-data.csv', test_size=0.5, random_seed=0))