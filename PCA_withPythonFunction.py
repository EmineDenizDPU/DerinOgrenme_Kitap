import numpy as np
from sklearn.decomposition import PCA

# Örnek bir veri kümesi
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])

# PCA modeli oluştur
pca = PCA(n_components=2)  # İlgilenilen temel bileşen sayısı (2 boyuta indirgeniyor)

# Veriyi PCA modeline uygula ve dönüşüm yap
transformed_data = pca.fit_transform(data)

# Dönüştürülmüş veriyi kullan
print("Dönüştürülmüş Veri:")
print(transformed_data)

