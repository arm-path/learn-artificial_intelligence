import random

#  Обучающая выборка, идеальное изображение цифр от 0-9.
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

# Список всех цифр от 0-9 в едином массиве.
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

tema = 5  # К какой цифре обучаем.
n_sensor = 15  # Количество сенсоров.
weights = [0 for i in range(n_sensor)]  # Обнуление весов.


# Функция определяет число 5.
def perceptron(Sensor):
    b = 7  # Порог функции активации.
    s = 0  # Начальное значение суммы.
    for i in range(n_sensor):  # Цикл суммирования сигналов от сенсоров.
        s += int(Sensor[i]) * weights[i]
    if s >= b:
        return True  # Сумма превысила порог.
    else:
        return False  # Сумма не превысила порог.


# Функция уменьшения значения весов.
def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1


# Функция увеличения значения весов.
def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] += 1


# Тренировка сети.
n = 100000  # Количество уроков.
for i in range(n):
    j = random.randint(0, 9)  # Генерируем случайное число j от 0-9.
    r = perceptron(nums[j])  # Результат обращения к сумматору с пороговой функцией активации.

    if j != tema:
        if r:  # Если сумматор ошибся, и сказал ДА (это пятерка).
            decrease(nums[j])
    else:
        if not r:  # Если сумматор ошибся, и сказал НЕТ (это не пятерка), когда это пять.
            increase(nums[tema])

print(weights)

# Тестовая выборка (различные варианты изображения цифры пять)
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

# Прогон по тестовой выборке
print('Узнал 5? ', perceptron(num51))
print('Узнал 5? ', perceptron(num52))
print('Узнал 5? ', perceptron(num53))
print('Узнал 5? ', perceptron(num54))
print('Узнал 5? ', perceptron(num55))
print('Узнал 5? ', perceptron(num56))
