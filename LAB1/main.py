import numpy as np
import pandas as pd


def f(x):
    return np.exp(x) / (x + np.log(x))


def count_h(n, left, right):
    return (right - left) / n


def count_r(m, integral1, h1, integral2, h2):
    result = (integral2 - integral1) / (h1 ** m - h2 ** m)
    return result * h2 ** m


def count_q_with_middle_rectangles(n, left, right):
    h = count_h(n, left, right)
    q = 0

    for k in range(n):
        q += f(left + k * h + h / 2)

    return h * q


def count_q_with_simpson(n, left, right):
    h = count_h(n, left, right)
    result = f(left) + f(right)

    sum1 = 0
    for k in range(1, n):
        sum1 += f(left + k * h)

    result += (2 * sum1)

    sum2 = 0
    for k in range(n):
        sum2 += f(left + k * h + h / 2)

    result += (4 * sum2)
    return (h / 6) * result


def count_integral(n1, n2, e, left, right, func, m):
    q1 = func(n1, left, right)
    h1 = count_h(n1, left, right)
    q2 = func(n2, left, right)
    h2 = count_h(n2, left, right)
    r = count_r(m, q1, h1, q2, h2)
    data = []
    k = 1

    while True:
        if abs(r) <= e:
            return data, np.arange(1, k)
        else:
            k += 1
            n1, n2 = n2, n2 * 2
            q1 = q2
            h1 = count_h(n1, left, right)
            q2 = func(n2, left, right)
            h2 = count_h(n2, left, right)
            r = count_r(m, q1, h1, q2, h2)
            data.append([(n1, n2), (h1, h2), q1, q2, r])


def integral(left, right):
    t = [-(3 / 5) ** (1 / 2), 0, (3 / 5) ** (1 / 2)]
    y = [5 / 9, 8 / 9, 5 / 9]
    result = 0

    for i in range(len(t)):
        result += f((left + right) / 2 + t[i] * (right - left) / 2) * y[i]

    return result * ((right - left) / 2)


if __name__ == '__main__':
    name_columns = ['Разбиения', 'Шаг', 'Интеграл1', 'Интеграл2', 'Погрешность']
    a, b = 2, 3
    eps = 10 ** (-7)
    m_s = [2, 4]
    functions = [count_q_with_middle_rectangles, count_q_with_simpson]

    info, index = count_integral(2, 4, eps, 2, 3, functions[0], m_s[0])
    info1, index1 = count_integral(2, 4, eps, 2, 3, functions[1], m_s[1])

    print('КФ средних прямоугольников')
    print(pd.DataFrame(info, index, name_columns), end='\n' * 2)
    print(f'Значение интеграла, вычисленное по КФ СП:  {info[-1][3]}', end='\n' * 2)

    print('КФ Симпсона')
    print(pd.DataFrame(info1, index1, name_columns), end='\n' * 2)
    print(f'Значение интеграла, вычисленное по КФ Симпсона:  {info1[-1][3]}', end='\n' * 2)

    print(f"Значение интеграла, вычисленное с помощью КФ НАСТ: {integral(a, b)}")
