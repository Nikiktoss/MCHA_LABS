from math import *
import numpy as np


def count_f1(x, y):
    return sin(x - 0.5 * y) + x + y ** 2 - 5


def count_f2(x, y):
    return 16 * x ** 2 - y ** 2 - 4


def count_partial_derivative_f1_x(x, y):
    return cos(x - 0.5 * y) + 1


def count_partial_derivative_f1_y(x, y):
    return 2 * y - 0.5 * cos(x - 0.5 * y)


def count_partial_derivative_f2_x(x):
    return 32 * x


def count_partial_derivative_f2_y(y):
    return -2 * y


def newtons_method(array_x):
    eps = 10 ** (-6)
    matrix = np.array([[count_partial_derivative_f1_x(array_x[0], array_x[1]),
                        count_partial_derivative_f1_y(array_x[0], array_x[1])],
                       [count_partial_derivative_f2_x(array_x[0]), count_partial_derivative_f2_y(array_x[1])]])

    f = np.array([-count_f1(array_x[0], array_x[1]), -count_f2(array_x[0], array_x[1])])
    delta_x = np.linalg.inv(matrix).dot(f)
    x1 = array_x + delta_x
    number_of_iteration = 1
    print(f"{number_of_iteration}: {x1[0]}, {x1[1]}, {max(abs(delta_x))}")
    while max(abs(x1 - array_x)) > eps:
        array_x = x1
        matrix = np.array([[count_partial_derivative_f1_x(array_x[0], array_x[1]),
                            count_partial_derivative_f1_y(array_x[0], array_x[1])],
                           [count_partial_derivative_f2_x(array_x[0]), count_partial_derivative_f2_y(array_x[1])]])
        f = np.array([-count_f1(array_x[0], array_x[1]), -count_f2(array_x[0], array_x[1])])
        delta_x = np.linalg.inv(matrix).dot(f)
        x1 = array_x + delta_x
        number_of_iteration += 1
        print(f"{number_of_iteration}: {x1[0]}, {x1[1]}, {max(abs(delta_x))}")

    print(f"||f(x, y)|| = {max(abs(count_f1(x1[0], x1[1])), abs(count_f2(x1[0], x1[1])))}")


def secant_method(array_x0):
    eps = 10 ** (-6)
    array_x1 = np.array([-0.9, -2.9])
    matrix = np.array([[(count_f1(array_x1[0], array_x1[1]) - count_f1(array_x0[0], array_x1[1])) /
                        (array_x1[0] - array_x0[0]), (count_f1(array_x1[0], array_x1[1]) -
                                                      count_f1(array_x1[0], array_x0[1])) / (
                                    array_x1[1] - array_x0[1])],
                       [(count_f2(array_x1[0], array_x1[1]) - count_f2(array_x0[0], array_x1[1])) /
                        (array_x1[0] - array_x0[0]),
                        (count_f2(array_x1[0], array_x1[1]) - count_f2(array_x1[0], array_x0[1])) / (
                                    array_x1[1] - array_x0[1])]])

    f = np.array([-count_f1(array_x1[0], array_x1[1]), -count_f2(array_x1[0], array_x1[1])])
    delta_x = np.linalg.inv(matrix).dot(f)
    x1 = array_x1 + delta_x
    number_of_iterations = 1
    print(f"{number_of_iterations}: {x1[0]}, {x1[1]}, {max(abs(delta_x))}")
    while max(abs(x1 - array_x1)) > eps:
        array_x0 = array_x1
        array_x1 = x1
        matrix = np.array([[(count_f1(array_x1[0], array_x1[1]) - count_f1(array_x0[0], array_x1[1])) /
                            (array_x1[0] - array_x0[0]), (count_f1(array_x1[0], array_x1[1]) -
                                                          count_f1(array_x1[0], array_x0[1])) / (
                                    array_x1[1] - array_x0[1])],
                           [(count_f2(array_x1[0], array_x1[1]) - count_f2(array_x0[0], array_x1[1])) /
                            (array_x1[0] - array_x0[0]),
                            (count_f2(array_x1[0], array_x1[1]) - count_f2(array_x1[0], array_x0[1])) / (
                                    array_x1[1] - array_x0[1])]])

        f = np.array([-count_f1(array_x1[0], array_x1[1]), -count_f2(array_x1[0], array_x1[1])])
        delta_x = np.linalg.inv(matrix).dot(f)
        x1 = array_x1 + delta_x
        number_of_iterations += 1
        print(f"{number_of_iterations}: {x1[0]}, {x1[1]}, {max(abs(delta_x))}")

    print(f"||f(x, y)|| = {max(abs(count_f1(x1[0], x1[1])), abs(count_f2(x1[0], x1[1])))}")


def gauss_seidel_method(x, y):
    eps = 1e-6
    number_of_iteration = 0
    print(f"{number_of_iteration}: {x}, {y}, {max(abs(count_f1(x, y)), abs(count_f2(x, y)))}")
    while max(abs(count_f1(x, y)), abs(count_f2(x, y))) > eps:
        x_prev = x
        x = x_prev - count_f2(x_prev, y) / count_partial_derivative_f2_x(x_prev)
        while abs(x - x_prev) > eps:
            x_prev = x
            x = x_prev - count_f2(x_prev, y) / count_partial_derivative_f2_x(x_prev)

        y_prev = y
        y = y_prev - count_f1(x, y_prev) / count_partial_derivative_f1_y(x, y_prev)
        while abs(y - y_prev) > eps:
            y_prev = y
            y = y_prev - count_f1(x, y_prev) / count_partial_derivative_f1_y(x, y_prev)

        number_of_iteration += 1
        print(f"{number_of_iteration}: {x}, {y}, {max(abs(count_f1(x, y)), abs(count_f2(x, y)))}")

    print(f"||f(x, y)|| = {max(abs(count_f1(x, y)), abs(count_f2(x, y)))}")


x0 = np.array([-1, -3])
print("Newton method:")
newtons_method(x0)
print()
print("Secant method:")
secant_method(x0)
print()
print("Gauss Seidel method:")
print(gauss_seidel_method(x0[0], x0[1]))
