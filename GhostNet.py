import tensorflow as tf
from tensorflow.keras import layers, models
class GhostModule(tf.keras.layers.Layer):
    def __init__(self, output_channels, ratio=2, dw_kernel_size=3, strides=1):
        super(GhostModule, self).__init__()
        self.output_channels = output_channels
        self.ratio = ratio
        self.init_channels = int(output_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)

        self.primary_conv = layers.Conv2D(self.init_channels, kernel_size=1, strides=strides, padding='same', use_bias=False)
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=dw_kernel_size, strides=1, padding='same', use_bias=False)

    def call(self, inputs, training=False):
        x = self.primary_conv(inputs)
        x = self.depthwise_conv(x)
        return tf.concat([inputs, x], axis=-1)
class GhostNet(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.ghost1 = GhostModule(16)
        self.ghost2 = GhostModule(32, strides=2)
        self.ghost3 = GhostModule(64, strides=2)
        self.ghost4 = GhostModule(128, strides=2)
        self.ghost5 = GhostModule(256, strides=2)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ghost1(x)
        x = self.ghost2(x)
        x = self.ghost3(x)
        x = self.ghost4(x)
        x = self.ghost5(x)

        x = self.global_pool(x)
        x = self.dense(x)
        return x
# GhostNet modeli
model = GhostNet(num_classes=10)  # sınıf
model.build(input_shape=(None, 224, 224, 3))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

