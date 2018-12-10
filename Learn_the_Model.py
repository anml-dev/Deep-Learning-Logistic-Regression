import numpy as np
import h5py
from Logistic_Regression_Classifier import model
from Logistic_Regression_utils import load_dataset, process_dataset

# Loading the data (cat and non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x, test_set_x, num_px = process_dataset(train_set_x_orig, test_set_x_orig)

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.05, print_cost = True)

with h5py.File("mytestfile.hdf5", "w") as f:
    f.create_dataset("costs", data=d["costs"])
    f.create_dataset("Y_prediction_test", data=d["Y_prediction_test"])
    f.create_dataset("Y_prediction_train", data=d["Y_prediction_train"])
    f.create_dataset("w", data=d["w"])
    f.create_dataset("b", data=d["b"])
    f.create_dataset("learning_rate", data=d["learning_rate"])
    f.create_dataset("num_iterations", data=d["num_iterations"])
    f.create_dataset("num_px", data = num_px)
    f.create_dataset("classes", data = classes)
    
