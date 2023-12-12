import numpy as np
import matplotlib.pyplot as plt

# Veri seti oluştur
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Parametreler
learning_rate = 0.01
n_iterations = 1000
m = len(X)

# Model parametreleri
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((m, 1)), X]  # Bias eklenmiş X matrisi

# Stokastik Gradient Descent
for iteration in range(n_iterations):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta = theta - learning_rate * gradients

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
