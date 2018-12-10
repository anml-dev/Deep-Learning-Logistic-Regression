import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('datasets/test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def process_dataset(train_set_x_orig, test_set_x_orig):
    
    m_train = len(train_set_x_orig)
    m_test = len(test_set_x_orig)
    num_px = len(train_set_x_orig[0])

    # Reshape the training and test examples
    train_set_x_flatten = train_set_x_orig.reshape(num_px*num_px*3, m_train)
    test_set_x_flatten = test_set_x_orig.reshape(num_px*num_px*3, m_test)

    # Center and standartize your dataset. We divide every row of the dataset by 255 (the maximum value of a pixel channel)
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255

    return train_set_x, test_set_x, num_px
