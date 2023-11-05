import numpy as np

def median_normalization(data):
    # data: Normalizlenecek veri, özellikler sıralıdır (örneğin, bir özellik vektörü).

    median = np.median(data)

    normalized_data = data / median

    return normalized_data

# Örnek kullanım
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized_data = median_normalization(input_data)
print(normalized_data)

