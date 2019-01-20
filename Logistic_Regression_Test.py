import numpy as np
import h5py
import matplotlib.pyplot as plt
import skimage
from Logistic_Regression_Classifier import predict


with h5py.File('mytestfile.h5', 'r') as f:
   w = f['/w'][:]
   b = f['/b'].value
   num_px = f['/num_px'].value
   classes = f['/classes'][:]

# Test with your own image 
my_image = "Vito2.jpeg"    
#my_image = "Hydrangeas.jpg"

fname = "images/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, (num_px, num_px), mode='constant', anti_aliasing='None').reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(np.array(w), np.array(b), my_image)

plt.imshow(image)
plt.show()
print("\ny = " + str(np.squeeze(my_predicted_image)) + ", the algorithm predicts a \"" + np.array(classes)[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.\n")
