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


model= keras.Sequential()
#Evrişim 1, Havuzlama 1
model.add(layers.Conv2D(filters=96, kernel_size=(7,7),strides=(2,2),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
#Evrişim 2, Havuzlama 2
model.add(layers.Conv2D(filters=256,kernel_size=(5,5),strides=(2,2),activation='relu',padding='same'))
model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
#Evrişim 3, Evrişim 4, Evrişim 5
model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
model.add(layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
#Havuzlama 3
model.add(layers.MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
#Tam bağlı katman 1
model.add(Dense(4096,activation='relu'))
#Tam bağlı katman 2
model.add(Dense(4096, activation='relu'))
#Çıktı 
model.add(Dense(1000,activation='softmax'))


model.summary()

