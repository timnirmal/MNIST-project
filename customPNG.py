import cv as cv
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

# Load a model written in the HDF5 format
model_loaded = load_model('my_model')

# Convert the labels to one-hot vectors
# Converts a class vector (integers) to binary class matrix.
num_classes = 10

# get images from CustomIMG folder (1.png, 2.png, 3.png, 4.png, 5.png)
for i in range(1, 6):
    # img = cv.imread('CustomIMG/'+str(i)+'.png', 0)
    img = cv.imread(f'CustomIMG/{i}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    # img = cv.resize(img, (28, 28))
    # img = img.reshape(1, 28, 28, 1)
    # img = img.astype('float32')
    # img /= 255
    prediction = model_loaded.predict(img)
    result = np.argmax(prediction)
    print("Prediction: ", result)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

# TODO : NOTE that this is not tested yet.