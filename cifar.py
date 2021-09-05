import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from tensorflow.data.experimental import AUTOTUNE

from keras import models, layers
from keras.datasets import mnist, cifar10

from sklearn.model_selection import train_test_split

from datetime import datetime


plt.rcParams['figure.figsize'] = (16, 10)
plt.rc('font', size=15)

# Declare constants
BATCH_SIZE = 128
STEPS_PER_EPOCH = 0
VALIDATION_STEPS = 0
LR_SCHEDULE = 0.001
BUFFER_SIZE = 0
NUM_EPOCHS = 200

histories = {}

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('.', '/logs/', run_id)

# Function to create a callback function to set early stopping and tensorBoard
def get_callbacks(name, isLog=True):
  return [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
  ]

# set learning schedule
def set_lr_schedule(steps_per_epoch):
   return tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch*1000,
    decay_rate=1,
    staircase=False)

# show image with label
def show_image(image, label):
  plt.figure(figsize=(2, 2))
  img_size = image.numpy().shape[0]
  plt.imshow(image.numpy().reshape(img_size,img_size))
  plt.title(label)
  plt.axis('off')

# check image shape to make sure it is suitable for the model
def check_image_shape(data_batches):
  for images, labels in data_batches.take(1):
    print(images.shape)
    print(images[0].shape)

# save model history
def save_model_history(history, file_name):
  with open(file_name, 'wb') as file_out_handle:
    pickle.dump(history, file_out_handle)

# load model history
def load_model_history(file_name):
  print(file_name)
  with open(file_name, 'rb') as file_in_handle:
      return pickle.load(file_in_handle)
    
def load_cifar_data():
    global STEPS_PER_EPOCH, VALIDATION_STEPS

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    STEPS_PER_EPOCH = len(X_train)/BATCH_SIZE
    VALIDATION_STEPS = len(X_test)/BATCH_SIZE
    LR_SCHEDULE = set_lr_schedule(STEPS_PER_EPOCH)

    # Converting the pixels data to float type
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Standardizing (255 is the total number of pixels an image can have)
    X_train = X_train / 255
    X_test = X_test / 255 

    # One hot encoding the target class (labels)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test
  
  X_train, y_train, X_test, y_test  = load_cifar_data()

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
        .batch(batch_size=BATCH_SIZE)\
        .cache()\
        .repeat()\
        .prefetch(AUTOTUNE)
      
# Validation dataset
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
        .batch(batch_size=BATCH_SIZE)\
        .cache()\
        .repeat()\
        .prefetch(AUTOTUNE)

model_cifar = models.Sequential()

model_cifar.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model_cifar.add(layers.MaxPooling2D(pool_size=(2,2)))
model_cifar.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model_cifar.add(layers.MaxPooling2D(pool_size=(2,2)))
model_cifar.add(layers.Dropout(0.3))

model_cifar.add(layers.Flatten())
model_cifar.add(layers.Dense(128, activation='relu'))
model_cifar.add(layers.BatchNormalization())
model_cifar.add(layers.Dropout(0.5))
model_cifar.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

# Checking the model summary
model_cifar.summary()

model_cifar.compile(optimizer=tf.keras.optimizers.Adam(LR_SCHEDULE), loss='categorical_crossentropy', metrics=['accuracy'])

print('batch_size: {0} #Epochs: {1} steps_per_epoch: {2} validation_steps: {3}'.format(BATCH_SIZE, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS))

experiment_name = 'model_cifar'

model_history = model_cifar.fit(
    train_ds, 
    steps_per_epoch=STEPS_PER_EPOCH, 
    epochs=NUM_EPOCHS,
    validation_data=test_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=get_callbacks(experiment_name),
    verbose=2)

histories[experiment_name] = model_history.history

model_cifar.save(experiment_name + '.h5')
save_model_history(histories[experiment_name], experiment_name + '.pickle')
