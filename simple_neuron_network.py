import numpy as np


# Функция активации.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Описание класса Neuron.
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


# Описание класса Нейронная сеть из 3 слоев.
class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])  # Веса одинаковы для всех нейронов.
        bias = 0  # Смещение одинаково для всех нейронов.

        # Формируем сеть из трех нейронов.
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)  # Формируем выход Y1 из нейрона h1.
        out_h2 = self.h2.feedforward(x)  # Формируем выход Y2 из нейрона h2.
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print('Y=', network.feedforward(x))
