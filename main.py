# MLP - Multi-Layer Perceptron Model
import seaborn as sns
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.python import tf2
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.datasets import load_digits
import pydot

# Test for TensorFlow GPU support
if tf2.enabled():
    print("TensorFlow 2.0 GPU is supported")
else:
    print("TensorFlow 2.0 GPU is not supported")

print()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print()

print(tf.test.gpu_device_name())
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# X has the Inputs, Y has the Respective Outputs

print(x_train.shape)
print(x_test.shape)

# Split the dataset into training and validation sets
x_val  = x_train[50000:60000]
x_train = x_train[0:50000]

y_val  = y_train[50000:60000]
y_train = y_train[0:50000]


# Reshape the data to fit the model (28x28 -> 784) (2D -> 1D)
x_train = x_train.reshape(50000, 784)
x_val = x_val.reshape(10000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_val.shape)

# Convert to float32
# https://github.com/ageron/handson-ml/issues/265
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data (0-255) to (0-1) range (make it easier to work with, fast, accurate)
gray_scale = True
x_train /= gray_scale
x_val /= gray_scale
x_test /= gray_scale


print(y_train[0])
# Convert the labels to one-hot vectors
# Converts a class vector (integers) to binary class matrix.
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# In output, it will like (0,1,2,3,4,5,6,7,8,9)
# If y = 9 then binary representation will be like -> (0,0,0,0,0,0,0,0,0,1)
# where 9 is 1 and other are 0

print(y_train[0])

# MLP Model with tensorflow backend
# Image(url= "https://raw.githubusercontent.com/minsuk-heo/deeplearning/master/img/simple_mlp_mnist.png", width=500,
# height=250)

# Create the model
# model will be like ->
# Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output

# Putting in Placeholders
# Placeholders are variables that we will assign data to them later A placeholder is simply a
# variable that we will assign data to at a later date. It allows us to create our operations and build our
# computation graph, without needing the data. In TensorFlow terminology, we then feed data into the graph through
# these placeholders.

# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])

# MPL + Relu activation function + Dropout + Softmax + Batch Normalization


num_input = x_train.shape[1] # 784
num_classes = y_train.shape[1] # 10
batch_size = 128
nb_epochs = 20

# Initialize the model
model = Sequential()
# Add the first layer
model.add(Dense(512, input_shape=(784,), activation='relu'))
# Add Batch Normalization
model.add(tf.keras.layers.BatchNormalization())
# Add Dropout to prevent overfitting for the first layer
model.add(Dropout(0.5))

# Add the second layer
model.add(Dense(512, activation='relu'))
# Add Batch Normalization
model.add(tf.keras.layers.BatchNormalization())
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(10, activation='softmax'))

model.summary()


# Compile the model
# Loss function is the error between the model and the target
# Optimizer is the way we update the parameters to reduce the loss, often referred to as the learning rate
# Metrics is the quantity we want to measure about the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# Fit the model on the batches generated by the generator defined below
# The generator will loop indefinitely until we call the generator.close() method
# The generator will take the data from the x_train and y_train arrays and will loop over them
# The generator will yield a tuple that contains the next batch of data
# The generator will stop looping when the generator.close() method is called
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(x_val, y_val))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)


# Print the accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Plot the loss and accuracy curves for training and validation
# The first parameter is the name of the window
# The second parameter is the name of the subplot
plt.figure(1)


# Plot the loss for both training and validation
plt.subplot(211)
plt.xlim(0, nb_epochs)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


# Plot the accuracy for both training and validation
plt.subplot(212)
plt.xlim(0, nb_epochs)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')

# set postion of subplot(212)
plt.subplots_adjust(hspace=0.5)



# Show the figure
plt.show()

# save the figure
plt.savefig('simple_mlp_mnist.png')





# Plot the model
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Plot weights
# Plot the weights of the first layer
# The first layer is the first item in the model.layers array
# The weights are the first item in the layer.get_weights() array
# The weights are a numpy array
# The first item in the array is the bias
# The second item in the array is the kernel
# The third item in the array is the running mean
# The fourth item in the array is the running variance
plt.figure(2)


# Get the weights of the first layer
# layer_weights = model.layers[0].get_weights()
model_weights = model.get_weights()

layer1_weights = model_weights[0].flatten().reshape(-1,1)
layer1_bias = model_weights[1].flatten().reshape(-1,1)
layer1_running_mean = model_weights[2].flatten().reshape(-1,1)
layer1_running_var = model_weights[3].flatten().reshape(-1,1)


# Plot the weights
plt.plot(layer1_weights)
plt.plot(layer1_bias)
plt.plot(layer1_running_mean)
plt.plot(layer1_running_var)


fig = plt.figure(3)

h1_weights = model_weights[0].flatten().reshape(-1,1)
h2_weights = model_weights[2].flatten().reshape(-1,1)
out_weights = model_weights[4].flatten().reshape(-1,1)

plt.title('Weight matrices after model trained')
plt.subplot(1,3,1)
plt.title('Trained Model weights')
ax = sns.violinplot(h1_weights, color='b')
plt.title('weight matrices after model trained')
plt.xlabel('layer 1')

plt.title('Weight matrices after model trained')
plt.subplot(1,3,2)
plt.title('Trained Model weights')
ax = sns.violinplot(h2_weights, color='r')
plt.title('weight matrices after model trained')
plt.xlabel('layer 2')

plt.title('Weight matrices after model trained')
plt.subplot(1,3,3)
plt.title('Trained Model weights')
ax = sns.violinplot(out_weights, color='y')
plt.title('weight matrices after model trained')
plt.xlabel('Output layer')


# Show the figure
plt.show()


# Save the figure
plt.savefig('simple_mlp_mnist_weights_violinplot.png')


# Save the model
# Save the model to a HDF5 file.
# The '.h5' extension indicates that the model is written in HDF5 format.
# model.save_model('model.h5')
model.save("my_model")
save_model(model, "my_modelh5")










"""
# Input Layer
# Input Layer is the first layer of the neural network. It is the layer that takes the input data.
# The input data is the 28x28 grayscale image.


# Hidden Layer 1 
# Hidden Layer 1 is the second layer of the neural network. It takes the output of the first hidden 
# layer and passes it through a nonlinear activation function. The number of nodes in the hidden layer is set to 100.
# The activation function is the rectified linear unit (ReLU). The output of the hidden layer is then passed through 
# a dropout layer. The dropout layer is used to prevent overfitting. The dropout layer is used to randomly set a 
# fraction of the nodes to 0, thus preventing the network from overfitting. 

# Output Layer 
# Output Layer is the last layer of the neural network. It takes the output of the last hidden layer and
# passes it through a nonlinear activation function. The number of nodes in the output layer is set to 10. The 
# activation function is the softmax function. The output of the output layer is the predicted class. 

"""


"""


# Reshape to fit the model
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# Convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Normalize
X_train /= 255
X_test /= 255


# Convert to one-hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))


# Pooling


# Dropout

"""