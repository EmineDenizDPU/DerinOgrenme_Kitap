import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import os
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tqdm import tqdm
def add_conv_block(input_layer, filters):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=RandomNormal(stddev=0.02))(input_layer)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def add_upscale_block(input_layer, filters):
    x = layers.UpSampling2D()(input_layer)
    x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=RandomNormal(stddev=0.02))(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def add_downscale_block(input_layer, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02))(input_layer)
    x = layers.LeakyReLU(0.2)(x)
    return x
def build_generator(latent_dim, filters=512):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * filters, kernel_initializer=RandomNormal(stddev=0.02))(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, filters))(x)
    
    x = add_conv_block(x, filters)
    
    model = Model(inputs, x)
    return model
def build_discriminator(input_shape, filters=512):
    inputs = layers.Input(shape=input_shape)
    x = add_conv_block(inputs, filters)
    x = layers.Flatten()(x)
    x = layers.Dense(1, kernel_initializer=RandomNormal(stddev=0.02))(x)
    
    model = Model(inputs, x)
    return model
latent_dim = 100
initial_filters = 512

generator = build_generator(latent_dim, initial_filters)
discriminator = build_discriminator((4, 4, initial_filters))

generator_optimizer = Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = Adam(1e-4, beta_1=0.5, beta_2=0.9)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
@tf.function
def train_step(real_images, generator, discriminator, batch_size, latent_dim):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
def train_progan(dataset, epochs, batch_size, latent_dim):
    current_resolution = 4
    filters = initial_filters

    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, batch_size, latent_dim)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')
        
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_images = generator(noise, training=False)
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i+1)
                plt.imshow((generated_images[i] + 1) / 2)
                plt.axis('off')
            plt.show()
        
        if (epoch + 1) % 20 == 0:
            if current_resolution < 128:
                current_resolution *= 2
                filters //= 2
                
                generator.add(add_upscale_block(generator.output, filters))
                discriminator.add(add_downscale_block(discriminator.output, filters))

# Varsayalım ki elinizde bir görüntü veri kümesi var
# 'your_dataset' yerine gerçek veri kümenizi koymalısınız
dataset = tf.data.Dataset.from_tensor_slices(your_dataset).batch(32)
train_progan(dataset, epochs=100, batch_size=32, latent_dim=100)
