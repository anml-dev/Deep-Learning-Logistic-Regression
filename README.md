# Deep-Learning-Logistic-Regression in Python 3.7

The code builds an image-recognition algorithm that can classify pictures as cat or non-cat with about 70% accuracy.  The model is simple, it mimics a neural network of 1 layer, and it easily overfits the training data. No regularization is provided in this solution. The hyperparameters have values, as is given below:

 - _learning rate: 0.05_
 - _number of iterrations: 2000_

This results in train accuracy of about 72.25 and test accurasy of 68%.

The files are organized as follows:

* **Logistic_Regression_Utils.py** - contains utility functions for loading and preprocessing the datasets. The hdf5 files "datasets/train.h5" and "datasets/test.h5" contain the training and test sets of images. The function load_dataset() returns the train and test datasets, as well as the list of classes ("cat", "non-cat"). Each line of the train and test datasets is an array representing an image. Each image is of shape (number_of_pixels, number_of_pixels, 3). Here 3 is for the three RGB channels. The function process_dataset() first reshapes the datasets  (the images are flattened into single vectors of shape (num_px * num_px * 3, 1)), and then centers and standatdizes them (in this case it is enough to divide every row of the dataset by 255 - the maximum value of a pixel channel)

* **Logistic_Regression_Classifier.py** - contains functions for parameter initialization and computing the sigmoid function, implementation of forward and backward propagation (computes the cost function and its gradient), implementation of the gradient descent algorithm to optimize the weights and bias (update the parameters), compute predictions (use the learned parameters to predict the labels for a given set of excamples). At the end all functions are merged into a single model function. 

* **Learn_the_Model.py** - loads the datasets, learns the model and stores the learned parameters into a hdf5 file

* **Logistic_Regression_Test.py** - 

