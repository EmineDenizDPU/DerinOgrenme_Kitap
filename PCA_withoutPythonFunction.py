import numpy as np

# Veri hazırlığı (örnek veri)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])

# Verileri merkezileştirme (ortalama çıkar)
mean = np.mean(data, axis=0)
centered_data = data - mean

# Kovaryans matrisini hesaplama
covariance_matrix = np.cov(centered_data, rowvar=False)

# Kovaryans matrisini çözümleme
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Özdeğerleri büyükten küçüğe sıralama
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# İlgilenilen temel bileşen sayısı
n_components = 2

# İlk n_components özvektörü seçme
top_eigenvectors = eigenvectors[:, :n_components]

# Veriyi temel bileşenlerle çarpma
transformed_data = centered_data.dot(top_eigenvectors)

# Dönüştürülmüş veriyi kullan
print("Dönüştürülmüş Veri:")
print(transformed_data)

