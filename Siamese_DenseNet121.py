# -*- coding: utf-8 -*-
"""Siamese_DenseNet121.ipynb

"""
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="",
    project_name="",
    workspace="",
)

#import splitfolders
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import PIL
#import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
#from jupyterthemes import jtplot
#jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.integrate as integrate

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

img_size = (224, 224)
# create run-time augmentation on training and test dataset
# For training datagenerator,  we add normalization, shear angle, zooming range and horizontal flip
os.chdir('/work/netthinker/shared/flowers')

# Create an ImageDataGenerator for the training set
train_Original_datagen = ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    
)
train_Original_generator = train_Original_datagen.flow_from_directory(
    'train/',
    target_size=img_size,
    batch_size=5000,
    #batch_size=train_samples,
    class_mode='categorical',
)

# Create an ImageDataGenerator for the validation set
valid_Original_datagen = ImageDataGenerator(rescale=1./255)
valid_Original_generator = valid_Original_datagen.flow_from_directory(
    'validation/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)


X_Original_train, y_Original_train = next(train_Original_generator)

X_Original_test, y_Original_test = next(valid_Original_generator)

print(X_Original_train.shape)
print(y_Original_train.shape)
print(X_Original_test.shape)
print(y_Original_test.shape)

# For Cezanne-People Dataset
#os.chdir('/work/netthinker/shared/Converted_Images_Ahatsham/Converted-Images-flowers-CycleGAN-Cezanne-People-Ahatsham/')
#os.chdir('/work/netthinker/shared/Converted_Images_Ahatsham/Converted-Images-flowers-CycleGAN-Cezanne-Still-Life-With-Fruit-Ahatsham/')
os.chdir('/work/netthinker/shared/flowers-drawing/')

train_Cezanne_datagen = ImageDataGenerator(
    rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    
)
train_Cezanne_generator = train_Cezanne_datagen.flow_from_directory(
    'train/',
    target_size=img_size,
    batch_size=5000,
    #batch_size=train_samples,
    class_mode='categorical',
)

# Create an ImageDataGenerator for the validation set
valid_Cezanne_datagen = ImageDataGenerator(rescale=1./255)
valid_Cezanne_generator = valid_Cezanne_datagen.flow_from_directory(
    'validation/',
    target_size=img_size,
    batch_size=5000,
    class_mode='categorical',
)


X_Cezanne_train, y_Cezanne_train = next(train_Cezanne_generator)
X_Cezanne_test, y_Cezanne_test = next(valid_Cezanne_generator)

print(X_Cezanne_train.shape)
print(X_Cezanne_test.shape)

'''
Visualize an original image and its rotated version
'''

index = 400

image_original = X_Original_test[index]
image_Cezanne = X_Cezanne_test[index]
plt.figure(figsize=(2,2))
plt.imshow(image_original, interpolation="nearest", cmap="Greys")
plt.xticks(())
plt.yticks(())
plt.show()

plt.figure(figsize=(2,2))
plt.imshow(image_Cezanne, interpolation="nearest", cmap="Greys")
plt.xticks(())
plt.yticks(())
plt.show()

'''
Delete the TensorFlow graph before creating a new model, otherwise memory overflow will occur.
'''
tf.keras.backend.clear_session()

'''
To reproduce the same result by the model in each iteration, we use fixed seeds for random number generation. 
'''
np.random.seed(42)
tf.random.set_seed(42)

input_shape = (224, 224,3)

'''
Create the base network for learning shared representations
'''
base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=(224,224,3), weights='imagenet')
#base_model = tf.keras.applications.ResNet152(include_top=False, input_shape=(224,224,3), weights='imagenet')
head_model = base_model.output
head_model = tf.keras.Model(inputs = base_model.inputs, outputs = head_model)

input1 = tf.keras.layers.Input(shape=(224,224,3))
input2 = tf.keras.layers.Input(shape=(224,224,3))


embedding_network1 = head_model(input1)

embedding_network2 = head_model(input2)

'''
distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=None)([embedding_network1, embedding_network2])
normal_layer = tf.keras.layers.BatchNormalization()(distance)
distance1 = tf.keras.layers.Flatten()(normal_layer)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(distance1)
#siamese = tf.keras.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
'''

'''
Add a classification layer
'''

out1 = tf.keras.layers.Flatten()(embedding_network1)
output1 = tf.keras.layers.Dense(5, activation="softmax")(out1)
out2 = tf.keras.layers.Flatten()(embedding_network2)
output2 = tf.keras.layers.Dense(5, activation="softmax")(out2)

model_1 = tf.keras.models.Model(inputs=[input1, input2], outputs=[output1, output2])

'''
Display the model summary
'''
model_1.summary()

'''
Display the model graph
'''
tf.keras.utils.plot_model(model_1, show_shapes=True)

optimizer="adam"
model_1.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
               optimizer=optimizer,
               metrics=["accuracy"])
# 
# 
# '''
# Create a callback object of early stopping
# '''
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_dense_4_loss',
                                   min_delta=0, 
                                   patience=10, 
                                   verbose=1, 
                                   mode='auto',
                                   restore_best_weights=True)
 
# '''
# Train the model.
# We need to specify two types of labels for training and validation using lists.
# '''
history_1 = model_1.fit([X_Original_train, X_Cezanne_train], [y_Original_train, y_Cezanne_train], 
                     batch_size=16, 
                     epochs=100,
                     verbose=1,
                     validation_data=([X_Original_test, X_Cezanne_test], [y_Original_test, y_Cezanne_test])
                     #callbacks=[early_stopping_cb]
                     )

numOfEpochs = len(history_1.history['loss'])
print("Epochs: ", numOfEpochs)


y_test_predicted = model_1.predict([X_Original_test, X_Cezanne_test])
y_test_predicted_Original = np.argmax(y_test_predicted[0], axis=1) # get the label/index of the highest probability class
y_test_predicted_Cezanne = np.argmax(y_test_predicted[1], axis=1) # get the label/index of the highest probability class

'''
Prediction for training data
'''
y_train_predicted = model_1.predict([X_Original_train, X_Cezanne_train])
y_train_predicted_Original = np.argmax(y_train_predicted[0], axis=1) # get the label/index of the highest probability class
y_train_predicted_Cezanne = np.argmax(y_train_predicted[1], axis=1) # get the label/index of the highest probability class

'''
Get the integer labels for the Original data
'''
y_test_Original = np.argmax(y_Original_test, axis=1) # get the label/index of the highest probability class
y_train_Original = np.argmax(y_Original_train, axis=1) # get the label/index of the highest probability class


'''
Compute the train & test accuracies for the Original data
'''
train_accuracy_Original = accuracy_score(y_train_predicted_Original, y_train_Original)
test_accuracy_Original_1 = accuracy_score(y_test_predicted_Original, y_test_Original)


print("\nOriginal - Train Accuracy: ", train_accuracy_Original)
print("\nOriginal Classification - Test Accuracy: ", test_accuracy_Original_1)


print("\nTest Confusion Matrix (Original):")
print(confusion_matrix(y_test_Original, y_test_predicted_Original))

print("\nClassification Report (Original):")
print(classification_report(y_test_Original, y_test_predicted_Original))

y_Cezanne_test = np.argmax(y_Cezanne_test, axis =1)
y_Cezanne_train = np.argmax(y_Cezanne_train, axis =1)

train_accuracy_Cezanne = accuracy_score(y_train_predicted_Cezanne, y_Cezanne_train)
test_accuracy_Cezanne_1 = accuracy_score(y_test_predicted_Cezanne, y_Cezanne_test)

print("\nCezanne - Train Accuracy: ", train_accuracy_Cezanne)
print("\nCezanne Classification - Test Accuracy: ", test_accuracy_Cezanne_1)


print("\nTest Confusion Matrix (Cezanne):")
print(confusion_matrix(y_Cezanne_test, y_test_predicted_Cezanne))

print("\nClassification Report (Cezanne):")
print(classification_report(y_Cezanne_test, y_test_predicted_Cezanne))
