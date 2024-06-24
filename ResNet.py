import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Dense, Flatten, Add
from tensorflow.keras.models import Model
def identity_block(x, filters):
    filters1, filters2, filters3 = filters
    
    y = Conv2D(filters1, (1, 1))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters2, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters3, (1, 1))(y)
    y = BatchNormalization()(y)
    
    y = Add()([x, y])
    y = Activation('relu')(y)
    
    return y
def conv_block(x, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    
    y = Conv2D(filters1, (1, 1), strides=strides)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters2, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(filters3, (1, 1))(y)
    y = BatchNormalization()(y)
    
    shortcut = Conv2D(filters3, (1, 1), strides=strides)(x)
    shortcut = BatchNormalization()(shortcut)
    
    y = Add()([y, shortcut])
    y = Activation('relu')(y)
    
    return y
def ResNet50(input_shape=(224, 224, 3), classes=1000):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2))(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Residual Blocks
    x = conv_block(x, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])
    
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    
    x = conv_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    
    x = AveragePooling2D((7, 7))(x)
    
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=x)
    return model
# Modeli oluştur
resnet_model = ResNet50()

resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Model özeti
resnet_model.summary()

