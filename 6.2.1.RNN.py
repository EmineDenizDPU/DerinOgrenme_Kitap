import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('D:/archive/weatherHistory.csv')

temperature_data = data['Temperature (C)'].values

def create_dataset(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

# Parametreler
seq_length = 10
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 100
batch_size = 16
learning_rate = 0.01

x, y = create_dataset(temperature_data, seq_length)
x = np.reshape(x, (x.shape[0], x.shape[1], input_size))
y = np.reshape(y, (y.shape[0], output_size))

# Model 
model = Sequential()
model.add(SimpleRNN(hidden_size, activation='tanh', input_shape=(seq_length, input_size)))
model.add(Dense(output_size))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
model.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1)
predicted = model.predict(x)
# Sonuçlar
plt.plot(range(len(temperature_data)), temperature_data, label='Gerçek Veri')
plt.plot(range(seq_length, len(predicted) + seq_length), predicted, label='Tahmin')
plt.legend()
plt.show()
