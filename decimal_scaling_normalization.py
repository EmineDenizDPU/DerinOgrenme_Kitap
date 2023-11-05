import numpy as np

def decimal_scaling_normalization(data, scale=1):
    # data: Normalizlenecek veri, özellikler sıralıdır (örneğin, bir özellik vektörü).
    # scale: Ölçekleme faktörü (10 üzeri kaç basamak hassasiyetle ölçekleneceği).

    normalized_data = data / (10 ** scale)

    return normalized_data

# Örnek kullanım
input_data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
scaled_data = decimal_scaling_normalization(input_data, scale=2)
print(scaled_data)
