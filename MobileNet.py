import tensorflow as tf
from tensorflow.keras import layers, models
def depthwise_separable_conv(x, filters, kernel_size, strides=1):
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x
def build_mobilenet(input_shape=(224, 224, 3), num_classes=1000, alpha=1.0):
    input_tensor = tf.keras.Input(shape=input_shape)

    # Initial Convolution Block
    x = layers.Conv2D(int(32 * alpha), 3, strides=2, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Depthwise separable convolutions
    x = depthwise_separable_conv(x, int(64 * alpha), 3)
    x = depthwise_separable_conv(x, int(128 * alpha), 3, strides=2)
    x = depthwise_separable_conv(x, int(128 * alpha), 3)

    x = depthwise_separable_conv(x, int(256 * alpha), 3, strides=2)
    x = depthwise_separable_conv(x, int(256 * alpha), 3)

    x = depthwise_separable_conv(x, int(512 * alpha), 3, strides=2)

    for _ in range(5):
        x = depthwise_separable_conv(x, int(512 * alpha), 3)

    x = depthwise_separable_conv(x, int(1024 * alpha), 3, strides=2)
    x = depthwise_separable_conv(x, int(1024 * alpha), 3)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_tensor, outputs=x)
    return model
# Model oluşturma
mobilenet_model = build_mobilenet()
mobilenet_model.summary()

