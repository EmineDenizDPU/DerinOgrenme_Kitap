from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNIST veri setini yükleme
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize et
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# WGAN modelini oluşturma
def build_wgan(generator, critic):
    z = Input(shape=(100,))
    img = generator(z)
    critic.trainable = False
    valid = critic(img)
    return Model(z, valid)

# Wasserstein kaybı hesaplama fonksiyonu
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Critic (Diskriminatör) modelini oluşturma
def build_critic(img_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# Generator modelini oluşturma
def build_generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 64, input_dim=100))
    model.add(Reshape((7, 7, 64)))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding="same"))
    model.add(Activation("tanh"))
    return model

# WGAN'ı eğitme fonksiyonu
def train_wgan(generator, critic, combined, epochs, batch_size, sample_interval):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        for _ in range(x_train.shape[0] // batch_size):
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = generator.predict(noise)

            valid = -np.ones((half_batch, 1))
            fake = np.ones((half_batch, 1))

            d_loss_real = critic.train_on_batch(imgs, valid)
            d_loss_fake = critic.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            noise = np.random.normal(0, 1, (batch_size, 100))
            valid_gan = -np.ones((batch_size, 1))
            g_loss = combined.train_on_batch(noise, valid_gan)

        print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss))

        if epoch % sample_interval == 0:
            sample_images(epoch, generator)

# Modeli oluştur
generator = build_generator()
critic = build_critic((28, 28, 1))

# WGAN'ı ve birleştirilmiş modeli oluştur
optimizer = RMSprop(lr=0.00005)
critic.compile(loss=wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

generator.compile(loss=wasserstein_loss, optimizer=optimizer)

z = Input(shape=(100,))
img = generator(z)
critic.trainable = False
valid = critic(img)
combined = Model(z, valid)
combined.compile(loss=wasserstein_loss, optimizer=optimizer)

# WGAN'ı eğit
train_wgan(generator, critic, combined, epochs=30000, batch_size=64, sample_interval=1000)
