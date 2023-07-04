import matplotlib.pyplot as plt


class Solver:
    def __init__(self, h, t, a=0, b=1):
        self.h = h
        self.t = t
        self.a = a
        self.b = b

    def __get_size(self):
        n1 = int((self.b - self.a) / self.h)
        n2 = int((self.b - self.a) / self.t)

        return n1, n2

    def create_grid(self):
        n1, n2 = self.__get_size()
        xi = [0] * (n1 + 1)
        ti = [0] * (n2 + 1)

        for i in range(len(xi)):
            xi[i] = self.a + i * self.h

        for i in range(len(ti)):
            ti[i] = self.a + i * self.t

        return xi, ti

    @staticmethod
    def u(x, t):
        return (x ** 2 + 1) / (t + 1)

    def phi(self, x, t, sigma=1.0):
        if sigma == 1:
            return (-x ** 2 - 1) / (t ** 2 + 2 * t + 1) - 2 / (t + 1)
        return (-x ** 2 - 1) / ((t + self.t / 2) ** 2 + 2 * (t + self.t / 2) + 1) - 2 / ((t + self.t / 2) + 1)

    def solution(self):
        xi, ti = self.create_grid()
        sol = [[0] * len(xi) for _ in range(len(ti))]

        for j in range(len(ti)):
            for i in range(len(xi)):
                sol[j][i] = self.u(xi[i], ti[j])

        return sol

    def explicit_task(self):
        xi, ti = self.create_grid()
        u = [[0] * len(xi) for _ in range(len(ti))]

        for i in range(len(xi)):
            u[0][i] = self.u(xi[i], 0)

        for j in range(len(ti) - 1):
            u[j + 1][0] = self.u(0, ti[j + 1])

        for j in range(len(ti) - 1):
            u[j + 1][len(xi) - 1] = self.u(1, ti[j + 1])

        for j in range(len(ti) - 1):
            for i in range(1, len(xi) - 1):
                u[j + 1][i] = u[j][i] + (self.t / self.h ** 2) * (u[j][i + 1] - 2 * u[j][i] + u[j][i - 1]) + \
                              self.t * self.phi(xi[i], ti[j])

        return u

    def get_coeff(self, y, index, size, sigma=1.0):
        xi, ti = self.create_grid()
        a, b, c, f = [], [], [], []

        for i in range(size + 1):
            if i == 0:
                b.append(0)
                c.append(1)
                f.append(self.u(0, ti[index + 1]))
            elif i == size:
                c.append(1)
                a.append(0)
                f.append(self.u(1, ti[index + 1]))
            else:
                a.append(sigma / self.h ** 2)
                b.append(sigma / self.h ** 2)
                c.append(1 / self.t + (2 * sigma) / self.h ** 2)
                f.append(y[i] / self.t + self.phi(xi[i], ti[index], sigma) +
                         ((1 - sigma) / self.h ** 2) * (y[i + 1] - 2 * y[i] + y[i - 1]))

        return a, c, b, f

    def sweep_method(self, yj, index, size, sigma=1.0):
        a, c, b, f = self.get_coeff(yj, index, size, sigma)
        xi, ti = self.create_grid()

        alpha = [b[0] / c[0]]
        betta = [f[0] / c[0]]
        y = [0] * len(xi)

        for i in range(1, len(xi) - 1):
            alpha.append(b[i] / (c[i] - a[i - 1] * alpha[i - 1]))

        for i in range(1, len(xi)):
            betta.append((f[i] + a[i - 1] * betta[i - 1]) / (c[i] - a[i - 1] * alpha[i - 1]))

        y[len(xi) - 1] = betta[-1]
        for i in range(len(xi) - 2, -1, -1):
            y[i] = alpha[i] * y[i + 1] + betta[i]

        return y

    def implicit_task(self, sigma=1.0):
        xi, ti = self.create_grid()
        u = [[0] * (len(xi)) for _ in range(len(ti))]

        for i in range(len(xi)):
            u[0][i] = self.u(xi[i], 0)

        for j in range(len(ti) - 1):
            u[j + 1][0] = self.u(0, ti[j + 1])
            u[j + 1][-1] = self.u(1, ti[j + 1])
            tmp = self.sweep_method(u[j], j, len(xi) - 1, sigma)

            for k in range(1, len(xi) - 1):
                u[j + 1][k] = tmp[k]

        return u

    @staticmethod
    def count_accuracy(u, y):
        accuracy = []

        for j in range(len(u)):
            for i in range(len(u[0])):
                accuracy.append(abs(u[j][i] - y[j][i]))

        return max(accuracy)


def draw_plot(x, y, u):
    plt.plot(x, y)
    plt.scatter(x, u, c="r")
    plt.show()


sol1 = Solver(0.1, 0.1)
y1 = sol1.explicit_task()
u1 = sol1.solution()

# draw_plot(sol1.create_grid()[0], y1[-2], u1[-2])

print(f'Accuracy, when t=0.1, h=0.1')
print(sol1.count_accuracy(u1, y1), end="\n" * 2)

sol2 = Solver(0.1, 0.1 ** 2 / 2)
y2 = sol2.explicit_task()
u2 = sol2.solution()

# draw_plot(sol2.create_grid()[0], y2[-2], u2[-2])

print(f'Accuracy, when t={0.1 ** 2 / 2}, h=0.1')
print(sol2.count_accuracy(u2, y2), end="\n" * 2)

sol3 = Solver(0.1, 0.1)
y3 = sol3.implicit_task()
u3 = sol3.solution()
print(f'Accuracy, when t=0.1, h=0.1 and sigma=1')
print(sol3.count_accuracy(u3, y3), end="\n" * 2)

sol4 = Solver(0.1, 0.1)
y4 = sol4.implicit_task(0.5)
u4 = sol4.solution()
print(f'Accuracy, when t=0.1, h=0.1 and sigma=0.5')
print(sol4.count_accuracy(u4, y4))
