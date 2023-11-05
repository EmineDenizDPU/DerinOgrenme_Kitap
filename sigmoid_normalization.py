import numpy as np

def sigmoid_normalization(data, a=1, b=0):
    # data: Normalizlenecek veri, özellikler sıralıdır (örneğin, bir özellik vektörü).
    # a: Sigmoid işleminin hassasiyeti, büyük değerler dar bir aralık oluştururken küçük değerler daha geniş bir aralık oluşturur.
    # b: Yatay kaydırma (shift) değeri.

    sigmoid_normalized_data = 1 / (1 + np.exp(-a * (data - b)))

    return sigmoid_normalized_data

# Örnek kullanım
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized_data = sigmoid_normalization(input_data)
print(normalized_data)

