from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

# Örnek veri kümesi (sınıflar arasında ayrımı göstermek için)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([0, 0, 0, 1, 1, 1])

# LDA modelini oluşturun ve uygulayın
lda = LinearDiscriminantAnalysis(n_components=1)  # İlgilenilen bileşen sayısı
X_lda = lda.fit_transform(X, y)

# Dönüştürülmüş veriyi kullanın
print("Dönüştürülmüş Veri:")
print(X_lda)
