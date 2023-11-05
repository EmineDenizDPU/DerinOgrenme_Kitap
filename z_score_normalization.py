import numpy as np

def z_score_normalization(data):
    # data: Normalizlenecek veri, özellikler sıralıdır (örneğin, bir özellik vektörü).

    mean = np.mean(data)
    std_deviation = np.std(data)

    normalized_data = (data - mean) / std_deviation

    return normalized_data

# Örnek kullanım
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized_data = z_score_normalization(input_data)
print(normalized_data)
