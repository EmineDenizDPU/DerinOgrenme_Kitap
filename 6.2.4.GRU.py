import tensorflow as tf
# Veri
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  

# Model 
model = tf.keras.Sequential([
    tf.keras.layers.GRU(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.GRU(64, return_sequences=False),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
# Model performansı
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
