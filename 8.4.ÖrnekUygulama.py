import tensorflow as tf
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Veri seti
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url, extract=True)
# Dosya yolu
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# Veri ön işleme
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
# Önceden eğitilmiş VGG16 modelini yükleyin ve son katmanları çıkarın
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
# Yeni model oluşturun ve önceden eğitilmiş modelin katmanlarını ekleyin
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# Modeli eğitin
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
# Fine Tuning: Bazı katmanları eğitilebilir hale getirin
for layer in base_model.layers[-4:]:  # Son 4 katmanı eğitilebilir yapın
    layer.trainable = True

# Modeli yeniden derleyin (daha düşük bir öğrenme oranı kullanarak)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
# Modeli yeniden eğitin
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

