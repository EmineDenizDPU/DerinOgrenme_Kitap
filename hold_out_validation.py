from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Örnek bir sınıflandırma modeli, kullanacağınız modele göre değiştirilebilir

# Veri setinizi ve etiketlerinizi yükleyin (örneğin, X ve y olarak adlandıralım)
# X, özellik matrisi
# y, etiket vektörü

# Örnek veri yükleme kodu
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target

# Veriyi eğitim ve test setlerine bölmek için train_test_split fonksiyonunu kullanın
test_size = 0.2  # Test setinin oranı (örneğin, verinin %20'si test seti olarak ayrılacak)
random_state = 42  # Rastgele sayı üretimi için kullanılan seed değeri

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Modelinizi oluşturun (örnek olarak RandomForestClassifier kullanıldı)
model = RandomForestClassifier(n_estimators=100)

# Modeli eğitin
model.fit(X_train, y_train)

# Test seti üzerinde modelin performansını değerlendirin
accuracy = model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))
