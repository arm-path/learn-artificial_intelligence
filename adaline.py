import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка из интернета массива из 150 элементов.
# Загрузка их в объект DataFrame и печать.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
# print('Массив: ')
# print(df.to_string())

# Выборка 100 элементов ( Столбец 4 - названия цветков)
# Загрузка его в одномерный массив Y и печать. Y - правильные значения, при входных данных.
y = df.iloc[0:100, 4].values  # ['Iris-setosa', ...,  'Iris-versicolor', ...]
# print('Значения Y в виде названия цветков:')
# print(y)

# Преобразование названий цветков (Столбец 4)
# В одномерный массив (вектор) из -1 и 1
y = np.where(y == 'Iris-setosa', -1, 1)  # [-1, ... 1, ...]
# print('Значения Y в виде -1 и 1:')
# print(y)

# Выборка из объекта 100 элементов (Столбец 0 и столбец 2)
# Загрузка его в массив x и печать.
# Двумерная матрица, состоящая из 100 - строк, 2 - столбца.
x = df.iloc[0:100, [0, 2]].values  # [[5.1 1.4] [4.9 1.4] ... [... ...] ... [5.1 3. ] [5.7 4.1]]
# print('Значения X:')
# print(x)

# Формирование параметров значений для вывода на график.

# Первые 50 элементов (Строки 0-50, столбцы 0, 1)
plt.scatter(x[0:50, 0], x[0:50, 1], color='red', marker='o', label='цетинистый')

# Следующие 50 элементов (Строки 50-100, столбцы 0, 1)
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формирование названия осей и вывод графика на экран.
plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.show()


# Описание класса Perseptron.
class Perceptron:
    """ Классификатор на основе персептрона """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # Темп обучения (между 0.0 и 1.0)
        self.n_iter = n_iter  # Проходы по тренировочному набору данных.

    """ Обучение """

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])  # [0. 0. 0.]
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):  # zip(x, y) - [(array([5.1, 1.4]), -1),..., (array([5.7, 4.1]), 1)]
                update = self.eta * (target - self.predict(xi))  # Шаг обучения * Ошибка сети.
                self.w_[1:] += update * xi  # x.shape[1] | TODO: w_i+1 = w_i + update * x_i * rate
                self.w_[0] += update  # x_i = 1

                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    """ расчитать чистый вход - Сумматор """

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    """ Вернуть метку класса, после единичного скачка """

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


# Тренировка
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)  # Обучение.

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

# Визуализация границы решений.
from matplotlib.colors import ListedColormap


def plot_decision_regions(x, y, classifier, resolution=0.02):  # classifier - Perceptron
    # Настроить генератор маркеров и палитру.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения.
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

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


# Адаптивный линейный нейрон.
class AdaptiveLinearNeuron:
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate  # Темп обучения.
        self.niter = niter  # Количество проходов - эпох.

    def fit(self, x, y):
        # Обучение нейронной сети
        self.weight = np.zeros(1 + x.shape[1])  # [0. 0. 0.]
        self.cost = []

        for i in range(self.niter):
            output = self.net_input(x)  # Ответ нейронной сети.
            errors = y - output  # Вектор величин ошибок.
            self.weight[1:] += self.rate * x.T.dot(errors)  # [0.465 1.398] | TODO: w_i+1 = w_i + update * x_i * rate
            self.weight[0] += self.rate * errors.sum()  # x_i = 1
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, x):       # Вычисление чистого входного сигнала
        return np.dot(x, self.weight[1:]) + self.weight[0]

    def activation(self, x):
        # Вычислительная линейная активация.
        return self.net_input(x)

    def predict(self, x):
        # Вернуть метку класса после единичного скачка.
        return np.where(self.activation(x) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# learning rate = 0.01
aln1 = AdaptiveLinearNeuron(0.01, 10).fit(x, y)
ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Сумма квадратичных ошибок)')
ax[0].set_title('ADALINE. Темп обучения 0.01')

# learning rate = 0.0001
aln2 = AdaptiveLinearNeuron(0.0001, 10).fit(x, y)
ax[1].plot(range(1, len(aln2.cost) + 1), np.log10(aln2.cost), marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('log(Сумма квадратичных ошибок)')
ax[1].set_title('ADALINE. Темп обучения 0.0001')
plt.show()

x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# x_std [[-0.5810659  -1.01435952] ... [ 0.35866332  0.85894465]]

aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(x_std, y)

# Нарисовать картину.
plot_decision_regions(x_std, y, classifier=aln)
plt.title('ADALINE (Градиентный спуск)')
plt.xlabel('Длина чашелистника (стандартизованная)')
plt.ylabel('Длина лепестка (стандартизованная)')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')
plt.show()
