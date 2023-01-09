import numpy as np


an = [[0, 0, 0], [2 / 3, 0, 0], [-1 / 3, 1, 0]]
bn = [1 / 4, 2 / 4, 1 / 4]
cn = [0, 2 / 3, 2 / 3]
m_trapezia = 2
m_runge_kutta = 3


class CauchyTask:
    eps = 10 ** (-6)

    def __init__(self, h, a=1, b=2, u0=1):
        self.a = a
        self.b = b
        self.h = h
        self.u0 = u0

    @staticmethod
    def f(t, u):
        return (-u / t) * np.log(u / t)

    def t(self):
        n = int((self.b - self.a) / self.h)
        result = [self.a + i * self.h for i in range(n + 1)]
        return result

    @staticmethod
    def u(t):
        return t * np.exp((1 - t) / t)

    def new_f(self, yj1, yj, tj1, tj):
        return yj1 - (self.h / 2) * self.f(tj1, yj1) - yj - (self.h / 2) * self.f(tj, yj)

    def derivative_new_f(self, yj1, tj1):
        return 1 + (self.h / 2) * ((1 / tj1) * np.log(yj1 / tj1) + (1 / tj1))

    def newton_method(self, yj, tj, tj1):
        temp = yj
        y_res = temp - self.new_f(temp, yj, tj1, tj) / self.derivative_new_f(temp, tj1)
        while abs(y_res - temp) > self.eps:
            temp = y_res
            y_res = temp - self.new_f(temp, yj, tj1, tj) / self.derivative_new_f(temp, tj1)

        return y_res

    def implicit_trapezoid_method(self):
        ts = self.t()
        n = int((self.b - self.a) / self.h)
        y0 = [self.u0]

        for i in range(n):
            y0.append(self.newton_method(y0[i], ts[i], ts[i + 1]))

        return y0

    def k(self, coef_a, coef_c, yj, tj):
        k = [self.f(tj, yj)]
        for i in range(1, len(coef_c)):
            summary = yj
            for j in range(i):
                summary += (self.h * coef_a[i][j] * k[j])
            k.append(self.f(tj + coef_c[i] * self.h, summary))
        return k

    def runge_kutta(self, coef_a, coef_b, coef_c):
        ts = self.t()
        n = int((self.b - self.a) / self.h)
        y0 = [self.u0]

        for i in range(n):
            summary = 0
            ks = self.k(coef_a, coef_c, y0[i], ts[i])
            for j in range(len(coef_b)):
                summary += coef_b[j] * ks[j]
            summary *= self.h
            summary += y0[i]
            y0.append(summary)

        return y0

    @staticmethod
    def count_accuracy(u, y):
        result = []
        for i in range(len(u)):
            result.append(np.abs(u[i] - y[i]))
        return max(result)

    @staticmethod
    def count_runge_accuracy(y1, y2, m):
        result = []
        for i in range(len(y1)):
            result.append(np.abs(y1[i] - y2[2 * i]))
        return max(result) / (2 ** m - 1)

    @classmethod
    def show_result(cls, h, *args):
        solver = CauchyTask(h)
        x = solver.t()
        u_result = [solver.u(i) for i in x]
        print(f"Точное решение задачи Коши: {u_result}")
        if args == tuple():
            y = solver.implicit_trapezoid_method()
            print(f"Приближённое решение задачи Коши: {y}")
        else:
            y = solver.runge_kutta(args[0], args[1], args[2])
            print(f"Приближённое решение задачи Коши: {y}")
        print(f"Погрешность равна: {cls.count_accuracy(u_result, y)}", end="\n" * 2)


sol1_1 = CauchyTask(0.1)
y1_1 = sol1_1.implicit_trapezoid_method()
sol1_2 = CauchyTask(0.05)
y1_2 = sol1_2.implicit_trapezoid_method()
CauchyTask.show_result(0.1)
CauchyTask.show_result(0.05)

print(f"Погрешность по правилу Рунге равна: {CauchyTask.count_runge_accuracy(y1_1, y1_2, m_trapezia)}", end="\n" * 2)

sol2_1 = CauchyTask(0.1)
y2_1 = sol2_1.runge_kutta(an, bn, cn)
sol2_2 = CauchyTask(0.05)
y2_2 = sol2_2.runge_kutta(an, bn, cn)
CauchyTask.show_result(0.1, an, bn, cn)
CauchyTask.show_result(0.05, an, bn, cn)

print(f"Погрешность по правилу Рунге равна: {CauchyTask.count_runge_accuracy(y2_1, y2_2, m_runge_kutta)}", end="\n" * 2)
