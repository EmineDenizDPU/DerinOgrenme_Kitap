import numpy as np
import matplotlib.pyplot as plt

# Veri seti oluştur
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Parametreler
learning_rate = 0.001
beta1 = 0.9  # Exponential decay rate for the first moment estimates
beta2 = 0.999  # Exponential decay rate for the second moment estimates
epsilon = 1e-8  # Smoothing term to avoid division by zero
n_iterations = 1000
m = len(X)

# Model parametreleri
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((m, 1)), X]  # Bias eklenmiş X matrisi

# Adam optimizasyonu
m_t = np.zeros_like(theta)  # First moment estimate
v_t = np.zeros_like(theta)  # Second moment estimate
t = 0

for iteration in range(n_iterations):
    t += 1
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    m_t = beta1 * m_t + (1 - beta1) * gradients
    v_t = beta2 * v_t + (1 - beta2) * gradients**2
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    theta = theta - learning_rate / (np.sqrt(v_hat) + epsilon) * m_hat

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
