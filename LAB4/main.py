an = [[0, 0], [2 / 3, 0]]
bn = [1 / 4, 3 / 4]
cn = [0, 2 / 3]


class RungeKutta:

    def __init__(self, a, b, h):
        self.a = a
        self.b = b
        self.h = h

    def t(self):
        n = int((self.b - self.a) / self.h)
        result = [self.a + i * self.h for i in range(n + 1)]
        return result

    @staticmethod
    def u(t):
        return 1 / t

    @staticmethod
    def f1(y2):
        return y2

    @staticmethod
    def f2(tj, y1, y2):
        return -tj * y2 - ((tj ** 2 - 2) / tj ** 2) * y1

    def k(self, tj, y1, y2, a, c):
        k = [[self.f1(y2), self.f2(tj, y1, y2)], []]
        y2_21 = y2 + self.h * a[1][0] * k[0][1]
        k[1].append(self.f1(y2_21))
        y1_22 = y1 + self.h * a[1][0] * k[0][0]
        k[1].append(self.f2(tj + c[1] * self.h, y1_22, y2_21))
        return k

    def runge_kutta(self, a, b, c):
        t_j = self.t()
        n = int((self.b - self.a) / self.h)
        u1 = [1]
        u2 = [-1]
        for i in range(n):
            summary_1 = 0
            summary_2 = 0
            ks = self.k(t_j[i], u1[i], u2[i], a, c)
            for j in range(len(b)):
                summary_1 += (b[j] * ks[j][0])
                summary_2 += (b[j] * ks[j][1])
            summary_1 *= self.h
            summary_2 *= self.h

            u1.append(u1[i] + summary_1)
            u2.append(u2[i] + summary_2)

        return u1

    @staticmethod
    def count_accuracy(u, y):
        result = []
        for i in range(len(u)):
            result.append(abs(u[i] - y[i]))
        return max(result)

    @staticmethod
    def count_runge_accuracy(y1, y2, m):
        result = []
        for i in range(len(y1)):
            result.append(abs(y1[i] - y2[2 * i]))
        return max(result) / (2 ** m - 1)

    @classmethod
    def show_result(cls, left, right, step,  a, b, c):
        rg = RungeKutta(left, right, step)
        u1 = rg.runge_kutta(a, b, c)
        ts = rg.t()
        result_function = [rg.u(t) for t in ts]
        print(f"Численное решение задачи Коши: {u1}")
        print(f"Точное решение: {result_function}")
        print(f"Погрешность: {cls.count_accuracy(result_function, u1)}", end="\n" * 2)


RungeKutta.show_result(1, 2, 0.2, an, bn, cn)
RungeKutta.show_result(1, 2, 0.1, an, bn, cn)

rg1 = RungeKutta(1, 2, 0.2)
u1 = rg1.runge_kutta(an, bn, cn)
rg2 = RungeKutta(1, 2, 0.1)
u2 = rg2.runge_kutta(an, bn, cn)

print(f"Погрешность по правилу Рунге: {RungeKutta.count_runge_accuracy(u1, u2, m=2)}")
