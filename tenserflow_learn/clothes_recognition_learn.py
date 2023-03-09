import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

""" Формирование данных """
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Тренировочный массив изображений: ', train_images.shape)
print('Тренировочный массив меток: ', len(train_labels))

print('Тестовый массив изображений: ', test_images.shape)
print('Тестовый массив меток: ', len(test_labels))

# Выводит первое тренировочное изображение.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# RGB(0-250) -> 0-1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Выводит первое тренировочное изображение.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Выводит первые 25 тренировочных изображений.
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

""" Формирование нейронной сети """
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Преобразует формат двумерного массива (28 на 28) в одномерный 784.
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

""" Обучение нейронной сети """
model.fit(train_images, train_labels, epochs=10)

""" Тестирования на проверочных данных """
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nТочность на проверочны данных: ', test_acc)

predictions = model.predict(test_images)  # Предсказания нейронной сети на тестовых наборах данных.
print('Вероятность предсказаний для первого рисунка', predictions[0])
print('Первое изображение, после обучения: ', np.argmax(predictions[0]))
print('Метка первого изображения: ', test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    # Функция отображает изображение, предсказание и правильный ответ сети.
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    # Функция отображает вероятности предсказаний.
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


""" Отображение предсказания нейронной сети по 0-элементу """
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

""" Отображение предсказания нейронной сети по 10-элементу """
i = 10
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

""" Отображение первых X тестовых изображений, предсказания нейронной сети и правильного ответа """
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
