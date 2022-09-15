import numpy as np
from math import *


def f(argument):
    return sin(2 * argument) * log(argument + 5, e)


def count_s(array_x, size):
    s = [0] * (2 * size + 1)
    for j in range(2 * size + 1):
        num = 0
        for k in range(len(array_x)):
            num += array_x[k] ** j
        s[j] = num
    return s


def count_m(arr_x, arr_f, number):
    m = [0] * (number + 1)
    for j in range(number + 1):
        num = 0
        for k in range(len(arr_x)):
            num += arr_f[k] * arr_x[k] ** j
        m[j] = num

    return m


def create_matrix(array_s, size):
    matrix = []
    for index in range(size + 1):
        temp = []
        for ind in range(size + 1):
            temp.append(array_s[ind + index])
        matrix.append(temp)
    return matrix


def count_q(array_c, arg):
    q = 0
    for index in range(len(array_c)):
        q += array_c[index] * arg ** index
    return q


def count_delta(arr_x, arr_c):
    delta = 0
    for index in range(len(arr_x)):
        delta += (f(arr_x[index]) - count_q(arr_c, arr_x[index])) ** 2
    return delta


if __name__ == '__main__':
    x = []
    start_number = -2
    while start_number < 2:
        x.append(start_number)
        start_number += 0.1
    x.append(2)

    fx = []
    for i in range(len(x)):
        fx.append(f(x[i]))
    b2 = np.array(count_m(x, fx, 2))
    A2 = np.array(create_matrix(count_s(x, 2), 2))
    c2 = np.linalg.solve(A2, b2)
    print("Coefficients si with n = 2: ")
    print(count_s(x, 2))
    print("Coefficients mi with n = 2: ")
    print(count_m(x, fx, 2))
    print(f"Coefficients ci with n = 2: {c2}")
    print(f"Delta with n = 2:  {count_delta(x, c2)}")
    print()

    b5 = np.array(count_m(x, fx, 5))
    A5 = np.array(create_matrix(count_s(x, 5), 5))
    c5 = np.linalg.solve(A5, b5)
    print("Coefficients si with n = 5: ")
    print(count_s(x, 5)[0: 5])
    print(count_s(x, 5)[5:])
    print("Coefficients mi with n = 5: ")
    print(count_m(x, fx, 5))
    print(f"Coefficients ci with n = 5: {c5}")
    print(f"Delta with n = 5:  {count_delta(x, c5)}")
