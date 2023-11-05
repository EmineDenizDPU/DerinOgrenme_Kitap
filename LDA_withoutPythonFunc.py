import numpy as np

# Örnek veri kümesi
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])  # Sınıf etiketleri (0 ve 1)

# Veriyi sınıflara göre ayırma
X_class_0 = X[y == 0]
X_class_1 = X[y == 1]

# Sınıf ortalamalarını hesaplama
mean_class_0 = np.mean(X_class_0, axis=0)
mean_class_1 = np.mean(X_class_1, axis=0)

# Sınıf içi dağılımları hesaplama
cov_class_0 = np.cov(X_class_0, rowvar=False)
cov_class_1 = np.cov(X_class_1, rowvar=False)

# Sınıflar arası kovaryans matrisi
S_B = np.outer((mean_class_0 - mean_class_1), (mean_class_0 - mean_class_1))

# Sınıf içi kovaryans matrisi
S_W = cov_class_0 + cov_class_1

# S_W'in tersini alıp S_B ile çarpma
S_W_inv = np.linalg.pinv(S_W)
eigenvalues, eigenvectors = np.linalg.eig(S_W_inv.dot(S_B))

# Özvektörleri sıralama
eigenvalue_index = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, eigenvalue_index]

# İlgilenilen bileşen sayısı
n_components = 1
top_eigenvectors = eigenvectors[:, :n_components]

# Veriyi dönüştürme
X_lda = X.dot(top_eigenvectors)

# Dönüştürülmüş veriyi kullan
print("Dönüştürülmüş Veri:")
print(X_lda)
