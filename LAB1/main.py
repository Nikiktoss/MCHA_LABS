import matplotlib.pyplot as plt


class Solver:
    def __init__(self, a, b, h):
        self.a = a
        self.b = b
        self.h = h

    def create_grid(self):
        n = int((self.b - self.a) / self.h)
        return [self.a + i * self.h for i in range(n + 1)]

    def ai(self, x):
        return 1 / self.h ** 2 - x ** 2 / (2 * self.h)

    def c0(self):
        return 1 / self.h + 2 + self.h / 2 * self.a - self.h * self.a ** 2

    def ci(self, x):
        return 2 / self.h ** 2 + x

    def cn(self):
        return 1 / self.h + self.h / 2 * self.b

    def an_b0(self):
        return 1 / self.h

    def bi(self, x):
        return 1 / self.h ** 2 + x ** 2 / (2 * self.h)

    def f0(self):
        return 4 + self.h / 2 * self.fi(self.a) - (self.h / 2) * 4 * self.a ** 2

    def fn(self):
        return -1 / 4 + self.h / 2 * self.fi(self.b) - (self.h / 8) * self.b ** 2

    @staticmethod
    def fi(x):
        return (3 * x ** 3 - 6) / x ** 4

    @staticmethod
    def u(x):
        return 1 / x ** 2

    def count_coefficients(self):
        a, c, b, f = [], [], [], []
        grid = self.create_grid()

        for i in range(len(grid)):
            if i == 0:
                c.append(self.c0())
                b.append(self.an_b0())
                f.append(self.f0())
            elif i == len(grid) - 1:
                a.append(self.an_b0())
                c.append(self.cn())
                f.append(self.fn())
            else:
                a.append(self.ai(grid[i]))
                c.append(self.ci(grid[i]))
                b.append(self.bi(grid[i]))
                f.append(self.fi(grid[i]))

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

    @staticmethod
    def count_value(y2h, yh):
        return 1 / 3 * max([abs(y2h[i] - yh[2 * i]) for i in range(len(y2h))])


def draw_plots(x, u, y):
    plt.plot(x[:2], y[:2])
    plt.plot(x[:2], u[:2])
    plt.xlim(1.0, 1.0005)
    plt.ylim(0.999, 1.0)
    plt.show()


sol1 = Solver(1, 2, 0.01)

y1 = sol1.sweep_method()
print(y1)

u1 = [sol1.u(x) for x in sol1.create_grid()]
print(u1)

print(f'Accuracy is {sol1.count_accuracy(u1, y1)}', end='\n' * 2)
# draw_plots(sol1.create_grid(), u1, y1)

sol2 = Solver(1, 2, 0.02)

y2 = sol2.sweep_method()
print(y2)

u2 = [sol2.u(x) for x in sol2.create_grid()]
print(u2)
print(f'Accuracy is {sol2.count_accuracy(u2, y2)}', end='\n' * 2)
# draw_plots(sol2.create_grid(), u2, y2)

print(f'Accuracy between y2h and yh is {sol1.count_value(y2, y1)}')
