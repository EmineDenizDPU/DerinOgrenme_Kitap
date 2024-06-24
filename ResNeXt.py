import tensorflow as tf
from tensorflow.keras import layers, models
def convolution_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
def residual_block(x, filters, cardinality=32):
    # Grouped convolution
    grouped_channels = filters // cardinality
    groups = []

    for i in range(cardinality):
        group = convolution_block(x, grouped_channels, kernel_size=1)
        group = convolution_block(group, grouped_channels, kernel_size=3)
        groups.append(group)

    x = layers.Concatenate()(groups)
    x = convolution_block(x, filters, kernel_size=1)

    # Shortcut connection
    shortcut = convolution_block(x, filters, kernel_size=1, strides=1)

    x = layers.Add()([x, shortcut])
    return x
def build_resnext(input_shape=(224, 224, 3), num_classes=1000, num_blocks=[3, 4, 6, 3], cardinality=32):
    input_tensor = tf.keras.Input(shape=input_shape)

    # Initial convolution
    x = convolution_block(input_tensor, filters=64, kernel_size=7, strides=2)

    # Residual blocks
    for i, num_blocks in enumerate(num_blocks):
        for j in range(num_blocks):
            x = residual_block(x, filters=64 * (2 ** i), cardinality=cardinality)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=x)
    return model
# Modeli oluşturma
resnext_model = build_resnext()
#Model özeti
resnext_model.summary()
