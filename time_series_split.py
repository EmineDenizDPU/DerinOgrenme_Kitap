import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor  # Örnek bir regresyon modeli, kullanacağınız modele göre değiştirilebilir

# Örnek bir zaman serisi veri seti oluşturun (örneğin, tarih ve değer sütunları içeren bir DataFrame)
# Bu örnekte, bir tarih sütunu olan ve bu tarihlerle ilişkilendirilmiş bir değer sütunu kullanalım
# Aşağıdaki örnek, bir DataFrame'i oluşturmak içindir ve sizin gerçek veri setinize uyarlanmalıdır.

data = {
    'Tarih': pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'),
    'Değer': range(365)
}

df = pd.DataFrame(data)

# Veriyi tarih sütununa göre sıralayın
df = df.sort_values(by='Tarih')

# TimeSeriesSplit kullanarak zaman serisi bölümleme yapın
n_splits = 5  # Kaç parçaya bölüneceği
tscv = TimeSeriesSplit(n_splits=n_splits)

# Modelinizi oluşturun (örnek olarak RandomForestRegressor kullanıldı)
model = RandomForestRegressor(n_estimators=100)

# Veriyi X (özellikler) ve y (etiketler) olarak ayırın
X = df[['Tarih']]  # Özellik olarak sadece tarih kullanıldı, sizin özelliklerinize göre ayarlayın
y = df['Değer']    # Etiket

# TimeSeriesSplit kullanarak bölünmüş veri seti üzerinde iterasyon yapın
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Modeli eğitin
    model.fit(X_train, y_train)

    # Modeli test seti üzerinde değerlendirin
    accuracy = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100))
