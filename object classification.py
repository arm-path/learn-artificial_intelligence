import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Описание класса Perceptron.
class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # Темп обучения (0 - 1).
        self.n_iter = n_iter  # Количество итерации (Уроков).

    """
    x - Тренировочные данные: массив, размерность - x[n_samples, n_features],
    где:
        n_samles - Число образцов.
        n_features - Число признаков.
    y - целевые значения. Массив размерность y[n_samples]
    """

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])  # w_ - Одномерный массив, веса после обучения.
        # x.shape = (100, 2). x.shape[1] = 2.
        self.errors_ = []  # errors_: список ошибок, классивикация в каждой эпохе.

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def n_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.n_input(x) >= 0.0, 1, -1)


def check_network(i):
    r2 = ppn.predict(i)
    if r2 == 1:
        print('Вид Iris Setosa')
    else:
        print('Вид Iris versicolor')


# Загрузка из интернета данных, и вывод их на печать, запись их в объект DataFrame.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Данные об ирисах')
# print(df.to_string())
# df.to_csv('')

# Выборка из DataFrame 100 строк (столбец 0 и столбец 2), загрузка их в массив x.
x = df.iloc[0:100, [0, 2]].values  # [[5.1 1.4] [4.9 1.4] ... [n m]]
print('Значения x-100')
# print(x)

# Выборка из DataFrame 100 строк (стобец 4, названия цветков.).
y = df.iloc[0:100, 4].values  # ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', ...]

# Преобразование названий цветков (столбец 4) в массив чисел -1 и 1.
y = np.where(y == 'Iris-setosa', -1, 1)  # [-1, -1, -1, -1, -1, ...]
print('Значение названий цветков в виде -1 и 1, Y-100')
# print(y)


# Отображение в графике.
# Первые 50 элементов обучающей выборки (Строки 0-50, столбцы 0, 1)
# x[0:50, 0] - Длина чашелистника - x, x[0:50, 1] - Длина лепестка - y.
plt.scatter(x[0:50, 0], x[0:50, 1], color='red', marker='o', label='щетинистый')

# Следующие 50 элементов обучающей выборки (Строки 50-100, столюцы 0,1)
# x[50:100, 0] - Длина чашелистника - x, x[50:100, 1] - Длина лепестка - y.
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование названия осей и вывод графика на экран.
plt.xlabel('Длина чашелистника')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()

# Тренировка (j, extybt)
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

# Проверяем сеть.
check_network([5.5, 1.6])
check_network([6.4, 4.5])

# Визуализация разделительной границы.
from matplotlib.colors import ListedColormap


def plot_decision_regions(x, y, classifier, resolution=0.02):
    # Настроить генератор маркеров и политру
    markers = ('s', 'x', 'o', '^')
    colors = ('red', 'blue', 'green', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения.
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1  # Длина чашелистника - х - коорд.
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # Длина лепестка - y - корд.

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Показать образцы классов.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


# Нарисовать картинку.
plot_decision_regions(x, y, classifier=ppn)
plt.xlabel('Длина чашелистника, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper left')
plt.show()
