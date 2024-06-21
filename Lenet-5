import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#Veri setini yükleme
(trainImages,trainLabels),(testImages,testLabels)=mnist.load_data()
trainImages = trainImages.reshape((60000, 28, 28, 1)).astype('float32') / 255
testImages = testImages.reshape((10000, 28, 28, 1)).astype('float32') / 255

trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

#MODEL
model = models.Sequential()

model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#DERLEME
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
model.fit(trainImages, trainLabels, epochs=5, batch_size=64, validation_split=0.2)

# Modeli değerlendirme
test_loss, test_acc = model.evaluate(testImages, testLabels)
print(f"Test accuracy: {test_acc}")

