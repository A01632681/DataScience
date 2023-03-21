import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Load the dataset CIFAR10 
dataset = tf.keras.datasets.cifar10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
(train_images0, train_labels0), (test_images0, test_labels0) = dataset.load_data()

# Reshape the data to fit the model
train_labels = train_labels0.reshape(-1)
test_labels = test_labels0.reshape(-1)

# Get the number of classes
num_classification_categories = len(class_names)
print('Number of classes=%d' % num_classification_categories)
     
# Plot the first 25 images from the training set and display the class name below each image 
# Only to know the dataset
plt.figure(figsize=(10,10))
for i in range(25):
    # define subplot
    plt.subplot(5,5,i+1)
    # plot raw pixel data
    plt.imshow(train_images0[i], cmap=plt.get_cmap('gray'))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if class_names != None:
        # Add a label underneath, if we have one...
        plt.xlabel(class_names[train_labels[i]])
plt.show()

# Normalize the data
test_images = (test_images0 / 255.0).astype(np.float32) # 10000 test patterns, shape 10000*28*28  
train_images = (train_images0 / 255.0).astype(np.float32) # 60000 train patterns, shape 60000*28*28

# Define the callbacks for each model
callbacks = [EarlyStopping(monitor='val_accuracy', patience=2)]

### --------------------------- CNN --------------------------- ###
# Initialize the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=16, kernel_size=3,activation='relu',padding="same"))
model.add(keras.layers.Conv2D(32, 3,activation='relu',padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=2))	
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(64, 3, activation='relu',padding="same"))
model.add(keras.layers.Conv2D(64, 3, activation='relu',padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(num_classification_categories, activation='softmax'))
model.build(input_shape=(None,) + train_images.shape[1:]) # Build the model
model.summary() # Get a summary of the model, how many layers, neurons and parameters

# Compile the model with the optimizer, loss function and metrics
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

# Train the model, use the callbacks and the validation data
history = model.fit(train_images, train_labels,
                batch_size=128,
                epochs=5,
                validation_data=(test_images, test_labels),
                callbacks=callbacks)

# Plot the accuracy and the loss for the training and the validation data
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Save the model to not have to train it again 
keras.models.save_model(model, "ModelCNN.h5",save_format='h5')

# ## --------------------------- RNN --------------------------- ##
# # Reshape the data to fit the model (RNN as we are using TimeDistributed layers)
train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2], train_images.shape[3])
test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2], test_images.shape[3])

# Initialize the model
model = keras.Sequential([
        keras.layers.TimeDistributed(keras.layers.Conv2D(16, 3, activation='relu', padding="same", input_shape=(1, 32, 32, 3))),
        keras.layers.TimeDistributed(keras.layers.Conv2D(32, 3, activation='relu')),
        keras.layers.TimeDistributed(keras.layers.Conv2D(64, 3, activation='relu')),
        keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
        keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.LSTM(128, activation='relu', return_sequences=False),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax")
])
model.build(input_shape=(None,) + train_images.shape[1:]) # Build the model
model.summary() # Get a summary of the model, how many layers, neurons and parameters

# Compile the model with the optimizer, loss function and metrics
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,  
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
# Train the model, use the callbacks and the validation data
history = model.fit(train_images, train_labels, 
                    batch_size=128, 
                    epochs=5, 
                    validation_data=(test_images, test_labels), 
                    callbacks=callbacks)

# Plot the accuracy and the loss for the training and the validation data
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Save the model to not have to train it again 
keras.models.save_model(model, "ModelRNN.h5",save_format='h5')

## --------------------------- Test on a random image and uploaded image--------------------------- ##
# This part of the code is to test the model on a random image from the test set and on an uploaded image
# Can run in other file using the saved model

model = keras.models.load_model('ModelRNN.h5') # Load the model

# Test on a random image from the test set
i=np.random.randint(len(test_images)-25)
plt.imshow(test_images0[i], cmap=plt.get_cmap('gray')) # Show the image
prediction = model(test_images[i:i+1])[0,:] # Get the prediction
prediction_class = np.argmax(prediction)
class_name = class_names[prediction_class] # Get the class name
true_label = test_labels[i] # Get the true label
plt.xlabel(class_name+" "+("CORRECT" if prediction_class==true_label else "WRONG\n{}".format(class_names[true_label]))) # Show the class name and if it is correct or not
plt.xticks([]); plt.yticks([]); plt.grid(False); plt.show()

# Test on an uploaded image
image = Image.open(r"C:\Users\Carlos\Downloads\car.jpeg")
image = image.resize((32, 32))
image_array = np.array(image)
image_array = image_array.reshape((1, 1, 32, 32, 3))
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)

img_color = cv2.imread(r"C:\Users\Carlos\Downloads\car.jpeg")
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
image_array = np.array(image)
image_array = image_array.reshape((1, 1, 32, 32, 3))
plt.imshow(img_color)
plt.axis("off")
plt.show()
test_images = (img_color / 255.0).astype(np.float32)
input_shape=((None,) + test_images.shape[1:])
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)
class_name = class_names[predicted_class]
print(class_name)