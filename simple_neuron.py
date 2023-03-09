import numpy as np


# Функция активации: f(x) = 1 / (1 + e ^ (-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Создание класса Neuron
class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def y(self, x):
        s = np.dot(self.w, x) + self.b  # Суммируем входы.
        return sigmoid(s)  # Обращение к функции активации.


xi = np.array([2, 3])  # Задание значений входам.
wi = np.array([0, 1])  # Веса входных сенсоров.
bias = 4  # Смещение b.
n = Neuron(wi, bias)
print('Y = ', n.y(xi))
