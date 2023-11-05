import numpy as np

def batch_normalization(input_data, epsilon=1e-5):
    # İnput_data: Normalleştirilecek veri, özellikler sıralıdır (örneğin, bir mini-batch).
    # epsilon: Sıfıra bölünmeyi önlemek için küçük bir değer ekler.

    # Her bir özelliğin ortalamasını ve varyansını hesapla
    mean = np.mean(input_data, axis=0)
    variance = np.var(input_data, axis=0)

    # Ortalamadan çıkar ve varyansa böl
    normalized_data = (input_data - mean) / np.sqrt(variance + epsilon)

    return normalized_data

# Örnek kullanım
input_data = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])

normalized_data = batch_normalization(input_data)
print(normalized_data)

