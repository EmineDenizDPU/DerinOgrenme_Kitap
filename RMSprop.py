import numpy as np
import matplotlib.pyplot as plt

# Veri seti oluştur
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Parametreler
learning_rate = 0.01
beta = 0.9  # Decay rate for the running average
epsilon = 1e-8  # Smoothing term to avoid division by zero
n_iterations = 1000
m = len(X)

# Model parametreleri
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((m, 1)), X]  # Bias eklenmiş X matrisi

# RMSprop optimizasyonu
accumulated_square_gradients = np.zeros_like(theta)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    accumulated_square_gradients = beta * accumulated_square_gradients + (1 - beta) * gradients**2
    theta = theta - learning_rate / (np.sqrt(accumulated_square_gradients) + epsilon) * gradients

# Eğitim sonrası theta değerleri
print("Eğitim sonrası theta değerleri:")
print(theta)

# Modeli test etmek için örnek bir giriş
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Bias eklenmiş X matrisi
y_predict = X_new_b.dot(theta)

# Sonuçları görselleştir
plt.plot(X, y, "b.")
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Tahmin")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
