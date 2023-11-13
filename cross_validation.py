from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier  # Örnek bir sınıflandırma modeli, kullanacağınız modele göre değiştirilebilir
import numpy as np

# Veri setinizi ve etiketlerinizi yükleyin (örneğin, X ve y olarak adlandıralım)
# X, özellik matrisi
# y, etiket vektörü

# Örnek veri yükleme kodu
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target

# K-fold cross-validation için KFold nesnesini oluşturun
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# n_splits: Kaç parçaya bölüneceği
# shuffle: Veriyi karıştırma
# random_state: Rastgele sayı üretimi için kullanılan seed değeri

# Modelinizi oluşturun (örnek olarak RandomForestClassifier kullanıldı)
model = RandomForestClassifier(n_estimators=100)

# cross_val_score kullanarak cross-validation'ı gerçekleştirin
# scoring parametresini, kullanılan modelin uygun bir performans metriği ile değiştirebilirsiniz
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Cross-validation sonuçlarını yazdırın
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

