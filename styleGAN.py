import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
def mapping_network(latent_dim, num_layers):
    inputs = layers.Input(shape=(latent_dim,))
    x = inputs

    for _ in range(num_layers):
        x = layers.Dense(latent_dim, activation='relu')(x)
    
    model = tf.keras.Model(inputs, x)
    return model
class AdaIN(layers.Layer):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def call(self, inputs):
        content, style = inputs
        mean_content, var_content = tf.nn.moments(content, [1, 2], keepdims=True)
        mean_style, var_style = tf.nn.moments(style, [1, 2], keepdims=True)
        std_content = tf.sqrt(var_content + 1e-6)
        std_style = tf.sqrt(var_style + 1e-6)
        normalized_content = (content - mean_content) / std_content
        return std_style * normalized_content + mean_style
def style_block(x, style, filters):
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN()([x, style])
    return x
def build_generator(latent_dim, num_styles):
    inputs = layers.Input(shape=(latent_dim,))
    style = mapping_network(latent_dim, num_styles)(inputs)
    
    x = layers.Dense(4*4*512)(style)
    x = layers.Reshape((4, 4, 512))(x)

    for i in range(5):
        filters = 512 // (2 ** i)
        style = mapping_network(latent_dim, num_styles)(inputs)
        x = layers.UpSampling2D()(x)
        x = style_block(x, style, filters)
    
    x = layers.Conv2D(3, kernel_size=1, padding='same', activation='tanh')(x)
    model = tf.keras.Model(inputs, x)
    return model
def build_discriminator():
    inputs = layers.Input(shape=(128, 128, 3))
    x = inputs

    for i in range(5):
        filters = 512 // (2 ** i)
        x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, x)
    return model
latent_dim = 512
num_styles = 8

generator = build_generator(latent_dim, num_styles)
discriminator = build_discriminator()

generator.summary()
discriminator.summary()

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
@tf.function
def train_step(real_images, batch_size):
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
def train(dataset, epochs, batch_size):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, batch_size)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')
        
        # Örnek görüntüleri kaydet
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_images = generator(noise, training=False)
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i+1)
                plt.imshow((generated_images[i] + 1) / 2)
                plt.axis('off')
            plt.show()

# Varsayalım ki elinizde bir görüntü veri kümesi var
# 'your_dataset' yerine gerçek veri kümenizi koymalısınız
dataset = tf.data.Dataset.from_tensor_slices(your_dataset).batch(32)
train(dataset, epochs=100, batch_size=32)
