import numpy as np
import math
import matplotlib.pyplot as plt


class Iuf:
    def __init__(self, a, b, lamb, eps):
        self.a = a
        self.b = b
        self.lamb = lamb
        self.eps = eps

    @staticmethod
    def accuracy(y, y2, m):
        result = 0
        for i in range(len(y)):
            tmp = abs(y2[0 + i * 2] - y[i])
            if tmp > result:
                result = tmp

        return result / (2 ** m - 1)

    @staticmethod
    def count_accuracy(y, u):
        result = 0
        for i in range(len(y)):
            tmp = abs(y[i] - u[i])
            if tmp > result:
                result = tmp
        return result

    def runge_rule(self, func, m):
        n1, n2 = 2, 4
        while True:
            y1 = func(n1)
            y2 = func(n2)
            acc = self.accuracy(y1, y2, m)
            if acc <= self.eps:
                print(acc)
                return y2
            else:
                n1 = n2
                n2 = n2 * 2

    def h(self, n):
        return (self.b - self.a) / n

    def x(self, n):
        return [self.a + i * self.h(n) for i in range(n + 1)]

    def show_result(self, y, u):
        print(f"y = {y}")
        print(f"u = {u}")
        print(f"Погрешность: {self.count_accuracy(y, u)}", end="\n" * 3)


class QuadratureMethod(Iuf):
    def __init__(self, a, b, lamb, eps):
        super().__init__(a, b, lamb, eps)

    @staticmethod
    def f(x):
        return np.log((2 + x) / (1 + x))

    @staticmethod
    def k(x, s):
        return (x + s) / (1 + x + s)

    def count_a(self, n):
        coefficients = []
        step = self.h(n)
        for i in range(n + 1):
            if i == 0 or i == n:
                coefficients.append(step / 3)
            elif i % 2 == 1:
                coefficients.append(4 * step / 3)
            else:
                coefficients.append(2 * step / 3)

        return coefficients

    def count_y(self, n):
        coefficients = self.count_a(n)
        x = self.x(n)
        matrix, fxi = [], []
        for i in range(n + 1):
            tmp = []
            for j in range(n + 1):
                if i == j:
                    tmp.append(1 - self.lamb * coefficients[i] * self.k(x[i], x[i]))
                else:
                    tmp.append(-self.lamb * coefficients[j] * self.k(x[i], x[j]))

            matrix.append(tmp)
            fxi.append(self.f(x[i]))

        return np.linalg.solve(matrix, fxi)


class SuccessiveApproximations(Iuf):
    def __init__(self, a, b, lamb, eps):
        super().__init__(a, b, lamb, eps)

    @staticmethod
    def f(x):
        return 5 * x ** 2 - 2 * math.sin(x ** 2) + (x ** 4) * math.sin(x ** 2) + 2 * (x ** 2) * math.cos(x ** 2)

    @staticmethod
    def k(x, s):
        return (x ** 3) * math.cos(x * s)

    def count_y(self, y, n):
        step = self.h(n)
        x = self.x(n)
        result = []
        for i in range(len(x)):
            tmp = self.lamb * (step / 2) * (self.k(x[i], x[0]) * y[0] + self.k(x[i], x[i]) * y[i]) + self.f(x[i])
            s = 0
            for j in range(1, i):
                s += (self.k(x[i], x[j]) * y[j])

            s *= (self.lamb * step)
            result.append(tmp + s)

        return result

    def count_y_with_accuracy(self, n):
        y = [0] * (n + 1)

        while True:
            y1 = self.count_y(y, n)

            if self.count_accuracy(y1, y) <= self.eps / 100:
                return y1
            else:
                y = y1


print("Метод квадратур для ИУФ-2")
qm = QuadratureMethod(0, 1, 1, 5 * 10 ** (-5))
y_res = qm.runge_rule(qm.count_y, 4)
x_res = qm.x(len(y_res) - 1)
u_res = [1] * len(y_res)
qm.show_result(y_res, u_res)
# plt.plot(x_res, u_res)
# plt.plot(x_res, y_res)
# plt.ylim(0.9999, 1.00000999)
# plt.show()


print("Метод последовательных приближений для ИУВ-2")
sa = SuccessiveApproximations(0, 1, -1 / 5, 5 * 10 ** (-5))
y_res = sa.runge_rule(sa.count_y_with_accuracy, 2)
x_res = sa.x(len(y_res) - 1)
u_res = [5 * elem ** 2 for elem in x_res]
sa.show_result(y_res, u_res)
# plt.plot(x_res, u_res, x_res, y_res)
# plt.show()
