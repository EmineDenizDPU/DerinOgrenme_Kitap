import numpy as np def min_max_normalization(data, a=0, b=1): 
# data: Normalizlenecek veri, özellikler sıralıdır (örneğin, bir özellik vektörü). 
# a: Yeni aralığın alt sınırı.
# b: Yeni aralığın üst sınırı. 
min_val = np.min(data) 
max_val = np.max(data) 

normalized_data = a + (data - min_val) * (b - a) / (max_val - min_val) 

return normalized_data 
# Örnek kullanım 
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) 
normalized_data = min_max_normalization(input_data) print(normalized_data)

