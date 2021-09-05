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

# Rescaling
def rescaling (image, is_reshape=False):
  x = tf.cast(image, tf.float32) / 255
  print(x.shape)
  if bool(is_reshape):
    x = np.expand_dims(x, axis=-1) # <--- add batch axis
    print(x.shape)
  return x

def convert (y):
  return to_categorical(y)

# Load data
def load_data(is_reshape=False):
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Set aside validation data
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

  X_train = rescaling(X_train, is_reshape)
  X_test = rescaling(X_test, is_reshape)
  X_val = rescaling(X_val, is_reshape)

  y_train = convert(y_train)
  y_test = convert(y_test)
  y_val = convert(y_val)

  print('Training size:', X_train.shape)
  print('Training label size:', y_train.shape)

  print('Validation size:', X_val.shape)
  print('Validation label size:', y_val.shape)

  print('Test size:', X_test.shape)
  print('Test label size:', y_test.shape)

  return X_train, y_train, X_test, y_test, X_val, y_val

def set_global_variables (batch_size, xTrain_size, xVal_size):
  global BATCH_SIZE, STEPS_PER_EPOCH, VALIDATION_STEPS, LR_SCHEDULE, BUFFER_SIZE
  
  BATCH_SIZE = batch_size
  STEPS_PER_EPOCH = xTrain_size/BATCH_SIZE
  VALIDATION_STEPS = xVal_size/BATCH_SIZE
  LR_SCHEDULE = set_lr_schedule(STEPS_PER_EPOCH)
  BUFFER_SIZE = xTrain_size

# set learning schedule
def set_lr_schedule(steps_per_epoch):
   return tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch*1000,
    decay_rate=1,
    staircase=False)

def input_pipelines(xTrain, yTrain, xVal, yVal, xTest, yTest, batchSize = BATCH_SIZE, train_cache = True):

  # Train dataset
  train_ds = tf.data.Dataset.from_tensor_slices((xTrain, yTrain))

  if bool(train_cache):
    train_ds = train_ds.cache()\
          .shuffle(buffer_size=BUFFER_SIZE)\
          .batch(batch_size=batchSize)\
          .prefetch(AUTOTUNE)
        
  # Validation dataset
  val_ds = tf.data.Dataset.from_tensor_slices((xVal, yVal))\
          .batch(batch_size=batchSize)\
          .cache()\
          .prefetch(AUTOTUNE)

  # Test dataset
  test_ds = tf.data.Dataset.from_tensor_slices((xTest, yTest))\
          .batch(batch_size=batchSize)\
          .cache()\
          .prefetch(AUTOTUNE)

  return train_ds, val_ds, test_ds



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

# load MNIST data for VGG16
X_train, y_train, X_test, y_test, X_val, y_val = load_data(is_reshape=True) 
set_global_variables (BATCH_SIZE, len(X_train), len(X_val))

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# pre-process images
IMAGE_SIZE = 32
def pre_process_image(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
  image = tf.image.grayscale_to_rgb(image)
  return image, label

# input pipelines
train_batches = train_ds.map(pre_process_image).batch(BATCH_SIZE).cache().repeat().prefetch(AUTOTUNE)
val_batches = val_ds.map(pre_process_image).batch(BATCH_SIZE).cache().repeat().prefetch(AUTOTUNE)
test_batches = test_ds.map(pre_process_image).batch(BATCH_SIZE).cache().repeat().prefetch(AUTOTUNE)

#-----------------------------------------------------
# VGG16 pretrained model on imagenet as a base model
conv_base_vgg = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

conv_base_vgg.summary()

# Construct the model
def make_model_vgg():
  model = models.Sequential()
  model.add(conv_base_vgg)
  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.keras.optimizers.Adam(LR_SCHEDULE), loss='categorical_crossentropy', metrics=['accuracy'])

  return model

#-------------------------------------------------------------

# without fine tune, pretrained base model weight is frozen

conv_base_vgg.trainable = False

model_vgg_noTune = make_model_vgg()
model_vgg_noTune.summary()

experiment_name = 'model_vgg_noTune'

print('batch_size: {0} #Epochs: {1} steps_per_epoch: {2} validation_steps: {3}'.format(BATCH_SIZE, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS))

tf.keras.backend.clear_session()
model_history = model_vgg_noTune.fit(
    train_batches, 
    steps_per_epoch=STEPS_PER_EPOCH, 
    epochs=3  #NUM_EPOCHS,
    validation_data=val_batches,
    validation_steps=VALIDATION_STEPS,
    callbacks=get_callbacks(experiment_name),
    verbose=2
)
histories[experiment_name] = model_history.history

# save model
model_vgg_noTune.save(experiment_name + '.h5')
save_model_history(histories[experiment_name], experiment_name + '.pickle')

#------------------------------------------------------------------
# conv_base_vgg.trainable = True

# set_trainable = False
# for layer in conv_base_vgg.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
        
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
      
# model_vgg_fineTune = make_model_vgg()

# experiment_name = 'model_vgg_fineTune'

# print('batch_size: {0} #Epochs: {1} steps_per_epoch: {2} validation_steps: {3}'.format(BATCH_SIZE, NUM_EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS))

# model_history = model_vgg_fineTune.fit(
#     train_batches, 
#     steps_per_epoch=STEPS_PER_EPOCH, 
#     epochs=NUM_EPOCHS,
#     validation_data=val_batches,
#     validation_steps=VALIDATION_STEPS,
#     callbacks=get_callbacks('model_vgg_fineTune'),
#     verbose=2
# )
# histories[experiment_name] = model_history.history

# model_vgg_fineTune.save(experiment_name + '.h5')
# save_model_history(histories[experiment_name], experiment_name + '.pickle')




