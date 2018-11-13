import numpy as np
from pkg.NN import NN

# набор учебных данных
X_train = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])
y_train = np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]]).T
# набор тестовых данных
X_test = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])
y_test = np.array([[0, 0, 1, 1]]).T

print("\nОтклонение")
n1 = NN(3, 1)
n1.train(X_train, y_train)
result1 = np.abs(n1.predict(X_test) - y_test)
print(result1)

print("\nОтклонение")
n2 = NN(3, 1)
n2.train(X_train, y_train, with_regulation=True)
result2 = np.abs(n2.predict(X_test) - y_test)
print(result2)

print("\nСравнение результатов")
print(result1 > result2)
