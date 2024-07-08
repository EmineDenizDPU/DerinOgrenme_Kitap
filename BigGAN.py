import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
class ConditionalBatchNorm(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(ConditionalBatchNorm, self).__init__(**kwargs)
        self.num_classes = num_classes
    
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(self.num_classes, input_shape[-1]),
                                     initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=(self.num_classes, input_shape[-1]),
                                    initializer='zeros', trainable=True, name='beta')
        self.bn = layers.BatchNormalization()
    
    def call(self, inputs, labels):
        gamma = tf.gather(self.gamma, labels)
        beta = tf.gather(self.beta, labels)
        return self.bn(inputs) * gamma + beta
def generator_block(input, labels, num_classes, filters, kernel_size=3, strides=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input)
    x = ConditionalBatchNorm(num_classes)(x, labels)
    x = layers.ReLU()(x)
    return x

def build_generator(latent_dim, num_classes):
    noise = layers.Input(shape=(latent_dim,))
    labels = layers.Input(shape=(), dtype=tf.int32)
    
    x = layers.Dense(4*4*512, use_bias=False)(noise)
    x = layers.Reshape((4, 4, 512))(x)
    x = ConditionalBatchNorm(num_classes)(x, labels)
    x = layers.ReLU()(x)
    
    x = layers.UpSampling2D()(x)
    x = generator_block(x, labels, num_classes, 256)
    
    x = layers.UpSampling2D()(x)
    x = generator_block(x, labels, num_classes, 128)
    
    x = layers.UpSampling2D()(x)
    x = generator_block(x, labels, num_classes, 64)
    
    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    
    model = tf.keras.Model([noise, labels], x)
    return model
def discriminator_block(input, filters, kernel_size=3, strides=1, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_discriminator(input_shape, num_classes):
    image = layers.Input(shape=input_shape)
    labels = layers.Input(shape=(), dtype=tf.int32)
    
    x = discriminator_block(image, 64)
    x = discriminator_block(x, 128, strides=2)
    x = discriminator_block(x, 256, strides=2)
    x = discriminator_block(x, 512, strides=2)
    
    x = layers.Flatten()(x)
    
    output = layers.Dense(1)(x)
    label_output = layers.Embedding(num_classes, 512)(labels)
    label_output = layers.Dense(tf.keras.backend.int_shape(x)[-1])(label_output)
    label_output = layers.Reshape((tf.keras.backend.int_shape(x)[-1],))(label_output)
    
    combined_output = layers.add([output, label_output])
    
    model = tf.keras.Model([image, labels], combined_output)
    return model
latent_dim = 128
num_classes = 1000
input_shape = (128, 128, 3)

generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(input_shape, num_classes)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
@tf.function
def train_step(real_images, real_labels, batch_size, latent_dim):
    noise = tf.random.normal([batch_size, latent_dim])
    fake_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, fake_labels], training=True)

        real_output = discriminator([real_images, real_labels], training=True)
        fake_output = discriminator([generated_images, fake_labels], training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
def train(dataset, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch, batch_size, latent_dim)

        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')
        
        # Örnek görüntüleri kaydet
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([batch_size, latent_dim])
            fake_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
            generated_images = generator([noise, fake_labels], training=False)
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i+1)
                plt.imshow((generated_images[i] + 1) / 2)
                plt.axis('off')
            plt.show()


dataset = tf.data.Dataset.from_tensor_slices((your_images, your_labels)).shuffle(10000).batch(batch_size)
train(dataset, epochs=100, batch_size=32, latent_dim=128)
