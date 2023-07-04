import matplotlib.pyplot as plt


class Solver:
    def __init__(self, a, b, h):
        self.a = a
        self.b = b
        self.h = h

    def create_grid(self):
        n = int((self.b - self.a) / self.h)
        return [self.a + i * self.h for i in range(n + 1)]

    @staticmethod
    def r(x):
        return x ** 2

    @staticmethod
    def q(x):
        return x

    @staticmethod
    def f(x):
        return (3 * x ** 3 - 6) / x ** 4

    def bi(self, x):
        return self.r(x) / 2

    def d(self, x):
        return (self.q(x + self.h / 2) + self.q(x - self.h / 2)) / 2

    def d_0_n(self, x):
        return self.q(x)

    def phi(self, x):
        return (self.f(x + self.h / 2) + self.f(x - self.h / 2)) / 2

    def ai(self, x):
        return 1 - self.h ** 2 / 6 * self.q(x - self.h / 2)

    def ai_coef(self, x):
        return self.ai(x) / self.h ** 2 - self.bi(x - self.h / 2) / self.h

    def bi_coef(self, x):
        return self.ai(x + self.h) / self.h ** 2 + self.bi(x + self.h / 2) / self.h

    def ci_coef(self, xi):
        return (self.ai(xi + self.h) + self.ai(xi)) / self.h ** 2 + self.bi(xi + self.h / 2) / self.h - \
               self.bi(xi - self.h / 2) / self.h + self.d(xi)

    def c0_coef(self, x):
        return (self.ai(x + self.h) + self.bi(x + self.h / 2) * self.h) / self.h + 2 + \
               self.h / 2 * self.d_0_n(x + self.h / 2)

    def b0_coef(self, x):
        return (self.ai(x + self.h) + self.bi(x + self.h / 2) * self.h) / self.h

    def f0_coef(self, x):
        return 4 + self.h / 2 * self.f(x + self.h / 2)

    def an_coef(self, x):
        return (self.ai(x) - self.bi(x - self.h / 2) * self.h) / self.h

    def cn_coef(self, x):
        return (self.ai(x) - self.bi(x - self.h / 2) * self.h) / self.h + self.h / 2 * self.d_0_n(x - self.h / 2)

    def fn_coef(self, x):
        return -1 / 4 + self.h / 2 * self.f(x - self.h / 2)

    @staticmethod
    def u(x):
        return 1 / x ** 2

    def count_coefficients(self):
        a, c, b, f = [], [], [], []
        grid = self.create_grid()

        for i in range(len(grid)):
            if i == 0:
                c.append(self.c0_coef(grid[i]))
                b.append(self.b0_coef(grid[i]))
                f.append(self.f0_coef(grid[i]))
            elif i == len(grid) - 1:
                a.append(self.an_coef(grid[i]))
                c.append(self.cn_coef(grid[i]))
                f.append(self.fn_coef(grid[i]))
            else:
                a.append(self.ai_coef(grid[i]))
                c.append(self.ci_coef(grid[i]))
                b.append(self.bi_coef(grid[i]))
                f.append(self.phi(grid[i]))

        return a, c, b, f

    def sweep_method(self):
        a, c, b, f = self.count_coefficients()
        grid = self.create_grid()

        alpha = [b[0] / c[0]]
        betta = [f[0] / c[0]]
        y = [0] * len(grid)

        for i in range(1, len(grid) - 1):
            alpha.append(b[i] / (c[i] - a[i - 1] * alpha[i - 1]))

        for i in range(1, len(grid)):
            betta.append((f[i] + a[i - 1] * betta[i - 1]) / (c[i] - a[i - 1] * alpha[i - 1]))

        y[len(grid) - 1] = betta[-1]
        for i in range(len(grid) - 2, -1, -1):
            y[i] = alpha[i] * y[i + 1] + betta[i]

        return y

    @staticmethod
    def count_accuracy(u, y):
        return max([abs(u[i] - y[i]) for i in range(len(y))])


def draw_plots(x, u, y):
    plt.plot(x[:2], y[:2])
    plt.plot(x[:2], u[:2])
    plt.xlim(1.0, 1.00005)
    plt.ylim(0.9999, 1.000)
    plt.show()


sol1 = Solver(1, 2, 0.01)

y1 = sol1.sweep_method()
print(y1)

u1 = [sol1.u(x) for x in sol1.create_grid()]
print(u1)

print(f'Accuracy is {sol1.count_accuracy(u1, y1)}', end='\n' * 2)
# draw_plots(sol1.create_grid(), u1, y1)

# plt.plot(sol1.create_grid(), u1)
# plt.show()
