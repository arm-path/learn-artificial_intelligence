from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

""" Получение данных """
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_tran, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

""" Стандартизация данных """
sc = StandardScaler()
sc.fit(X_tran)
X_tran_std = sc.transform(X_tran)
X_test_std = sc.transform(X_test)

""" Обучение нейронной сети - Персептрон """
ppn = Perceptron(eta0=0.1, random_state=0, max_iter=40)
ppn.fit(X_tran_std, y_train)

""" Тренировка, проверка результатов нейронной сети - Персептрон """
y_pred = ppn.predict(X_test_std)
print('Число ошибочно классифицированных образцов: %d' % (y_test != y_pred).sum())
print('Вероятность: %.2f' % accuracy_score(y_test, y_pred))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """ Графическое представление обученной нейронной сети """

    markers = ('s', 'x', 'o', '>', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Показать все образцы.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        # Выделить тестовые образцы.
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='0', alpha=1.0, linewidths=1, marker='.', s=55,
                        label='Тестовый набор')


X_combined_std = np.vstack((X_tran_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Длина лепестка (Стандартизованная)')
plt.ylabel('Ширина лепестка (Стандартизованная)')
plt.legend(loc='upper left')
plt.show()

""" Тренировка, проверка результатов нейронной сети - Логистическая регрессия """
Lr = LogisticRegression(C=1000.0, random_state=0)
Lr.fit(X_tran_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=Lr, test_idx=range(105, 150))
plt.xlabel('Длина лепестка (Стандартизованная)')
plt.ylabel('Ширина лепестка (Стандартизованная)')
plt.legend(loc='upper left')
plt.show()

X1 = np.array([[1.5, 1.5]])
X2 = np.array([[0.0, 0.0]])
X3 = np.array([[-1, -1]])
p1 = Lr.predict_proba(X1)
p2 = Lr.predict_proba(X2)
p3 = Lr.predict_proba(X3)
print(X1)
print(p1)
print(X2)
print(p2)
print(X3)
print(p3)
