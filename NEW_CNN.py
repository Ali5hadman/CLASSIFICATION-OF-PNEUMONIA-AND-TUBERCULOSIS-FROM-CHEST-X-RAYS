# for loading/processing the images  
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

from sklearn.metrics import accuracy_score,confusion_matrix # metrics error
from sklearn.model_selection import train_test_split # resampling method
# models 

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

import splitfolders
input_shape=(200,200)

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# Dataset folders 
print(os.chdir('/Users/franciscosanchez/Downloads/'))
dataset_dir = 'dataset4'
training_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')


labels = ["N",   #0
          "P",   #1
          "T",   #2
         ]



input_shape=(200,200)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator for training, validation, and test sets
train_data_gen = ImageDataGenerator(rescale = 1/255.)
valid_data_gen = ImageDataGenerator(rescale = 1/255.)
test_data_gen = ImageDataGenerator(rescale = 1/255.)

# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                               target_size=input_shape,
                                               color_mode='grayscale',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=32,
                                               shuffle=True,
                                               seed=seed)
valid_gen = train_data_gen.flow_from_directory(directory=validation_dir,
                                               target_size=input_shape,
                                               color_mode='grayscale',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=32,
                                               shuffle=False,
                                               seed=seed)
test_gen = train_data_gen.flow_from_directory(directory=test_dir,
                                              target_size=input_shape,
                                              color_mode='grayscale',
                                              classes=None, # can be set to labels
                                              class_mode='categorical',
                                              batch_size=32,
                                              shuffle=False,
                                              seed=seed)





input_shape = (200, 200,1)
epochs = 200


from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight = 'balanced',classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights= dict(enumerate(class_weights.flatten(), 0))
print(class_weights)




from keras import regularizers
bias_initializer = tf.keras.initializers.HeNormal()


def build_model(input_shape):

    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')
    x2 = tfkl.Conv2D(
        filters=16,
        kernel_size=5,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(input_layer)
    x = tfkl.BatchNormalization()(x2)
    x = tfkl.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPooling2D()(x)

    x1 = tfkl.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)




# x = tfkl.BatchNormalization()(x)
#     x = tfkl.MaxPooling2D()(x)
#     x = tfkl.Conv2D(
#             filters=256,
#             kernel_size=3,
#             strides=1,
#             padding = 'same',
#             use_bias=True,
#             bias_initializer = tfk.initializers.HeUniform(seed),
#             activation = 'relu',
#             kernel_initializer = tfk.initializers.HeUniform(seed),
#             kernel_regularizer=regularizers.l2(l=0.01),
#         )(x)
#     x = tfkl.BatchNormalization()(x)
#     x = tfkl.Conv2D(
#             filters=512,
#             kernel_size=3,
#             strides=1,
#             padding = 'same',
#             use_bias=True,
#             bias_initializer = tfk.initializers.HeUniform(seed),
#             activation = 'relu',
#             kernel_initializer = tfk.initializers.HeUniform(seed),
#             kernel_regularizer=regularizers.l2(l=0.01),
#         )(x)
#     x = tfkl.Conv2D(
#             filters=512,
#             kernel_size=3,
#             strides=1,
#             padding = 'same',
#             use_bias=True,
#             bias_initializer = tfk.initializers.HeUniform(seed),
#             activation = 'relu',
#             kernel_initializer = tfk.initializers.HeUniform(seed),
#             kernel_regularizer=regularizers.l2(l=0.01),
#         )(x)
#     x = tfkl.Conv2D(
#             filters=512,
#             kernel_size=3,
#             strides=1,
#             padding = 'same',
#             use_bias=True,
#             bias_initializer = tfk.initializers.HeUniform(seed),
#             activation = 'relu',
#             kernel_initializer = tfk.initializers.HeUniform(seed),
#             kernel_regularizer=regularizers.l2(l=0.01),
#         )(x)



    
    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPooling2D()(x)   

    x2 = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 

    x = tfkl.BatchNormalization()(x2)

    x = tfkl.MaxPooling2D()(x)
    
    x = tfkl.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 
    x = tfkl.MaxPooling2D()(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 
    #x = tfkl.MaxPooling2D()(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 



    x = tfkl.BatchNormalization()(x)

    y = tfkl.GlobalAveragePooling2D(name='gap')(x2)
    x = tfkl.Flatten()(x)
    x = tfkl.Concatenate()([x,y])
  
    classifier_layer = tfkl.Dense(units=2046, name='Classifier',
                                  kernel_initializer=tfk.initializers.HeUniform(seed), 
                                  use_bias=True,
                                  bias_initializer=tfk.initializers.HeUniform(seed),
                                  kernel_regularizer=regularizers.l2(l=0.01),
                                  activation='relu')(x)
    x = tfkl.Dropout(0.2, seed=seed)(classifier_layer)
    
    x = tfkl.Dense(units=1024, name='Classifier1',
                              kernel_initializer=tfk.initializers.HeUniform(seed), 
                              use_bias=True,
                              bias_initializer=tfk.initializers.HeUniform(seed),
                              kernel_regularizer=regularizers.l2(l=0.01),
                              activation='relu')(classifier_layer)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    
    x = tfkl.Dense(units=512, name='Classifier2',
                              kernel_initializer=tfk.initializers.HeUniform(seed), 
                              use_bias=True,
                              bias_initializer=tfk.initializers.HeUniform(seed),
                              kernel_regularizer=regularizers.l2(l=0.01),
                              activation='relu')(x)

    output_layer = tfkl.Dense(units=3, activation='softmax', 
                              kernel_initializer=tfk.initializers.GlorotUniform(seed), 
                              name='output_layer')(x)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model

# Utility function to create folders and callbacks for training
from datetime import datetime

def create_folders_and_callbacks(model_name):

  exps_dir = os.path.join('CNN_experiments')
  if not os.path.exists(exps_dir):
      os.makedirs(exps_dir)

  now = datetime.now().strftime('%b%d_%H-%M-%S')

  exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
  if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
      
  callbacks = []

  # Model checkpoint
  # ----------------
  ckpt_dir = os.path.join(exp_dir, 'ckpts')
  if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'), 
                                                     save_weights_only=True, # True to save only weights
                                                     save_best_only=False) # True to save only the best epoch 
  callbacks.append(ckpt_callback)

  # Visualize Learning on Tensorboard
  # ---------------------------------
  tb_dir = os.path.join(exp_dir, 'tb_logs')
  if not os.path.exists(tb_dir):
      os.makedirs(tb_dir)
  osstr='tensorboard --logdir /Users/franciscosanchez/Downloads/'+tb_dir
  print(osstr)
  # By default shows losses and metrics for both training and validation
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, 
                                               profile_batch=0,
                                               histogram_freq=1)  # if > 0 (epochs) shows weights histograms
  callbacks.append(tb_callback)

  # Early Stopping
  # --------------
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
  callbacks.append(es_callback)

  return callbacks






model=build_model(input_shape )
model.summary()
callbacks = create_folders_and_callbacks(model_name='CNN_Corino')



# Train the model
# Train the model
history = model.fit_generator(
    train_gen,
    epochs = 200,
    validation_data = valid_gen,
    class_weight = class_weights,
    callbacks = callbacks
)


model_metrics=model.evaluate(test_gen,return_dict=True)
model.save('CNN '+str(model_metrics['accuracy']))

f = plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
g = plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()






