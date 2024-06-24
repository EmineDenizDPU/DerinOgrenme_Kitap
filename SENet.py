import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense
#Squeeze and Excitation 
def squeeze_excite_block(input_tensor, filters, ratio=16):
    # Squeeze
    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(filters // ratio, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)

    # Excite
    x = tf.keras.layers.Reshape((1, 1, filters))(x)
    x = tf.keras.layers.Multiply()([input_tensor, x])

    return x
def senet_block(input_tensor, filters, ratio=16):
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x, filters, ratio)

    return x
def SENet(input_shape, num_classes, ratio=16):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = senet_block(x, 64, ratio)
    x = senet_block(x, 128, ratio)
    x = senet_block(x, 256, ratio)
    x = senet_block(x, 512, ratio)

    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output, name='senet')

    return model
#Model oluşturma
input_shape = (224, 224, 3)  
num_classes = 1000  
senet_model = SENet(input_shape, num_classes)
#Model özeti
senet_model.summary()

