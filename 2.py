# MLP - Multi-Layer Perceptron Model
import tensorflow as tf
from keras.datasets import mnist
from keras.models import load_model

# print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load a model written in the HDF5 format
model_loaded = load_model('my_model')

# Load the MNIST dataset
_, (x_test, y_test) = mnist.load_data()

# Reshape the data to fit the model (28x28 -> 784) (2D -> 1D)
x_test = x_test.reshape(10000, 784)

# Convert to float32
x_test = x_test.astype('float32')

# Normalize the data (0-255) to (0-1) range (make it easier to work with, fast, accurate)
gray_scale = True
x_test /= gray_scale

# Convert the labels to one-hot vectors
# Converts a class vector (integers) to binary class matrix.
num_classes = 10
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

################################################################################
# Predict the output
predictions = model_loaded.predict(x_test)
################################################################################

# Print the predictions
print("Prediction : ", predictions[0])

# Prediction to class
# Get the index of the highest probability
predicted_classes = tf.argmax(predictions, axis=1)
# Print the predicted classes
# print("Prediction Class : ", predicted_classes[0])
# Get value from the predicted_classes (binary to class)
print("Value must be : ", predicted_classes[0].numpy())

# y_test binary to class
# Get the index of the highest probability
y_test_classes = tf.argmax(y_test, axis=1)
print("True : ", y_test_classes[0].numpy())

################################################################################


# predicted_classes have the predicted values of all data
# y_test_classes have the true values of all respective data

# Calculate the number of correct predictions
correct_predictions = tf.equal(predicted_classes, y_test_classes)
# Calculate the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
# Print the accuracy
print("Accuracy : ", accuracy.numpy())


# Confusion Matrix
# Calculate the confusion matrix
confusion_matrix = tf.math.confusion_matrix(y_test_classes, predicted_classes)
# Print the confusion matrix
print("Confusion Matrix : ", confusion_matrix)


# Plot the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn


# Set the figure size
plt.figure(figsize=(10, 10))


# Plot the confusion matrix
sn.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="g")


# Set the labels
plt.xlabel("Predicted Class")
plt.ylabel("True Class")


# Set the title
plt.title("Confusion Matrix")


# Show the plot
plt.show()

################################################################################


# show image from the test set
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.show()

