from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, LeakyReLU, Add, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Diskteki veri seti yolu
dataset_path = '/path/to/dataset/'

# Veri setini yükleme fonksiyonu
def load_dataset(dataset_path, target_size=(256, 256)):
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        class_mode=None,
        shuffle=True,
        batch_size=1
    )
    return dataset

# Generator modeli oluşturma fonksiyonu
def build_generator():
    def conv2d(layer_input, filters, kernel_size=4, strides=2, padding='same'):
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)
        return d

    def residual_block(layer_input, filters=256):
        shortcut = layer_input
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization()(d)
        return Add()([shortcut, d])

    def deconv2d(layer_input, filters, kernel_size=4, strides=2, padding='same'):
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(u)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    # Giriş katmanı
    img = Input(shape=(256, 256, 3))

    # Dönüşüm (A->B)
    c1 = conv2d(img, 64)
    r = residual_block(c1)
    for _ in range(8):
        r = residual_block(r)
    c2 = deconv2d(r, 64)
    transform_AtoB = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh')(c2)

    # Geri dönüş (B->A)
    c3 = conv2d(transform_AtoB, 64)
    r = residual_block(c3)
    for _ in range(8):
        r = residual_block(r)
    c4 = deconv2d(r, 64)
    transform_BtoA = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh')(c4)

    return Model(img, [transform_AtoB, transform_BtoA])

# Diskriminatör modeli oluşturma fonksiyonu
def build_discriminator():
    def d_layer(layer_input, filters, kernel_size=4, strides=2, padding='same'):
        d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(layer_input)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    img = Input(shape=(256, 256, 3))

    # Dönüşüm (A->B) için discriminator
    d_A = d_layer(img, 64)
    d_A = d_layer(d_A, 128)
    d_A = d_layer(d_A, 256)
    d_A = d_layer(d_A, 512)

    # Geri dönüş (B->A) için discriminator
    d_B = d_layer(img, 64)
    d_B = d_layer(d_B, 128)
    d_B = d_layer(d_B, 256)
    d_B = d_layer(d_B, 512)

    validity_A = Conv2D(1, kernel_size=4, strides=1, padding='same')(d_A)
    validity_B = Conv2D(1, kernel_size=4, strides=1, padding='same')(d_B)

    return Model(img, [validity_A, validity_B])

# Ortak model için mae_loss fonksiyonu
def mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

# Ortak model için üreteç ve diskriminatörü oluşturma fonksiyonu
def build_cyclegan(generator_AtoB, generator_BtoA,
