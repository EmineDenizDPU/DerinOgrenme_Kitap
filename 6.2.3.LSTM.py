#dolar kur tahmini örneği
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veri
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame(index=dates, columns=['USD_TRY'])
df['USD_TRY'] = np.random.uniform(low=7.0, high=8.5, size=(100,))

# Veriyi normalize etme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

def create_dataset(data, timestep):
    X, Y = [], []
    for i in range(len(data) - timestep):
        X.append(data[i:i + timestep, 0])
        Y.append(data[i + timestep, 0])
    return np.array(X), np.array(Y)

timesteps = 10
X, Y = create_dataset(scaled_data, timesteps)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM modeli 
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(timesteps, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# tahmin 
last_data = scaled_data[-timesteps:]
last_data = np.reshape(last_data, (1, timesteps, 1))
predicted = model.predict(last_data)
predicted_price = scaler.inverse_transform(predicted)

print(f"Sonraki dolar kuru tahmini: {predicted_price[0][0]}")
