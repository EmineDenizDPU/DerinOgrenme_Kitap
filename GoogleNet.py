import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Dense
#Inception modülü
def inception_module(x, filters):
    conv1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    
    conv3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    conv3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv3x3)
    
    conv5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    conv5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv5x5)
    
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    maxpool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)
    
    inception = concatenate([conv1x1, conv3x3, conv5x5, maxpool], axis=-1)
    return inception

#GoogleNet modeli
def GoogleNet():
    input_layer = Input(shape=(224, 224, 3))
    
    # Initial Convolution
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception Modules
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = inception_module(x, [128, 192, 96, 64, 64, 64])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, [192, 208, 96, 64, 64, 64])
    x = inception_module(x, [160, 224, 112, 64, 64, 64])
    x = inception_module(x, [128, 256, 128, 64, 64, 64])
    x = inception_module(x, [112, 288, 144, 64, 64, 64])
    x = inception_module(x, [256, 320, 160, 128, 128, 128])
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1000, activation='softmax')(x)  
    
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    
    return model
# Modeli oluştur
googLeNet_model = GoogleNet()
# Model özeti
googLeNet_model.summary()

