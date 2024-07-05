import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, SimpleRNN

#Veri
X = np.random.rand(100, 10, 1)  
y = np.random.randint(2, size=100)  

# Model
model = Sequential()
model.add(Bidirectional(SimpleRNN(64), input_shape=(10, 1)))  
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=32)

loss, accuracy = model.evaluate(X, y)
print(f'Kayıp: {loss}, Doğruluk: {accuracy}')
