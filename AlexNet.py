import keras
from tensorflow import keras
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import keras.layers as layers 
import numpy as np

#Model oluşturma
model= keras.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11,11),strides=(4,4),activation='relu',input_shape=(227,227,3)))
model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))

model.add(layers.Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),activation='relu',padding='same'))
model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))

model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
model.add(layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
          
model.add(layers.Flatten())
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dropout(0.5))         
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1000,activation='softmax'))
sgd=SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.0005)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

batch_size=128
epochs=10

 model.summary()
