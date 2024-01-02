import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Veri setini yükle
(x_train, y_train), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize et
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Koşullu GAN modelini oluştur
def build_generator(z_dim, img_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(7 * 7 * 128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

def build_discriminator(img_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def build_cgan(generator, discriminator, z_dim, num_classes):
    z = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, z_dim)(label))
    model_input = multiply([z, label_embedding])
    img = generator(model_input)
    discriminator.trainable = False
    validity = discriminator([img, label])
    model = Model([z, label], validity)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

img_shape = (28, 28, 1)
z_dim = 100
num_classes = 10

generator = build_generator(z_dim, img_shape, num_classes)
discriminator = build_discriminator(img_shape, num_classes)
cgan = build_cgan(generator, discriminator, z_dim, num_classes)

def train_cgan(epochs=1, batch_size=128, sample_interval=50):
    batch_count = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train[idx]
            labels = labels.reshape(-1, 1)

            noise = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict([noise, labels])

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, z_dim))
            sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

            g_loss = cgan.train_on_batch([noise, sampled
