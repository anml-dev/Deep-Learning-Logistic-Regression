import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('datasets/train.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('datasets/test.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))
    
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

def process_dataset(train_x_orig, test_x_orig):

    num_px = train_x_orig.shape[1] # =height = width of a training image
       
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    return train_x, test_x, num_px