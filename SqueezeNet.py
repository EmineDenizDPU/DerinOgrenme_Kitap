import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, Dropout, Flatten, Dense
#Fire modül
def fire_module(x, squeeze_filters, expand_filters):
    squeezed = Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same')(x)
    expanded_1x1 = Conv2D(expand_filters, (1, 1), activation='relu', padding='same')(squeezed)
    expanded_3x3 = Conv2D(expand_filters, (3, 3), activation='relu', padding='same')(squeezed)
    return concatenate([expanded_1x1, expanded_3x3], axis=-1)
#SqueezeNet
def SqueezeNet(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='valid')(input_tensor)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)

    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output, name='squeezenet')

    return model
#Model oluşturma
input_shape = (224, 224, 3) 
num_classes = 1000  
squeeze_net_model = SqueezeNet(input_shape, num_classes)
# Model özeti
squeeze_net_model.summary()

