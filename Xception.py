import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def entry_flow(input_tensor):
    # Entry flow
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 728]:
        x = Activation('relu')(x)
        x = SeparableConv2D(size, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        residual = Conv2D(size, (1, 1), strides=(2, 2), padding='same')(previous_block_activation)
        x = Add()([x, residual])
        previous_block_activation = x  # Set aside next residual

    return x

def middle_flow(x, num_blocks=8):
    # Middle flow
    previous_block_activation = x

    for _ in range(num_blocks):
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, previous_block_activation])
        previous_block_activation = x

    return x

def exit_flow(x):
    # Exit flow
    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same')(previous_block_activation)
    x = Add()([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    return x

def Xception(input_shape=(299, 299, 3), num_classes=1000):
    input_tensor = Input(shape=input_shape)

    x = entry_flow(input_tensor)
    x = middle_flow(x)
    x = exit_flow(x)

    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)
    return model

model = Xception()
model.summary()
