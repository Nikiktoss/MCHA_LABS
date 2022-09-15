from math import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def my_function(number):
    return 4 ** number - 5 * number - 2


def count_phi_for_simple_iteration(number):
    return log(5 * number + 2, 4)


def derivative_of_the_function(number):
    return (4 ** number) * log(4) - 5


def method_of_half_division(number0, number1, eps):
    print("Method of half division: ")
    data = []
    list_of_pairs = []
    number_of_iteration = 0
    while number1 - number0 > 2 * eps:
        number_of_iteration += 1
        number2 = (number0 + number1) / 2
        data.append([number0, number1, my_function(number0), my_function(number1), number2, number1 - number0])

        list_of_pairs.append((number0, number1))
        if my_function(number2) * my_function(number0) < 0:
            number1 = number2
        else:
            number0 = number2
    return list_of_pairs[-1], data, np.arange(1, number_of_iteration + 1)


def method_of_simple_iteration(x, eps):
    print("Method of simple iteration: ")
    x_res = count_phi_for_simple_iteration(x)
    data = []
    number_of_iteration = 1
    data.append([x_res, abs(x_res - x)])
    while abs(x_res - x) > eps:
        x = x_res
        x_res = count_phi_for_simple_iteration(x)
        number_of_iteration += 1
        data.append([x_res, abs(x_res - x)])

    return data, np.arange(1, number_of_iteration + 1)


def newton_method(x, eps):
    print("The method of Newton: ")
    data = []
    x_res = x - my_function(x) / derivative_of_the_function(x)
    number_of_iteration = 1
    data.append([x_res, abs(x_res - x)])
    while abs(x_res - x) > eps:
        x = x_res
        x_res = x - my_function(x) / derivative_of_the_function(x)
        number_of_iteration += 1
        data.append([x_res, abs(x_res - x)])

    return data, np.arange(1, number_of_iteration + 1)


def steffensen_method(x, eps):
    print("Method of Steffensen: ")
    numerator = x * count_phi_for_simple_iteration(count_phi_for_simple_iteration(x)) - \
        count_phi_for_simple_iteration(x) ** 2
    denominator = count_phi_for_simple_iteration(count_phi_for_simple_iteration(x)) - \
        2 * count_phi_for_simple_iteration(x) + x
    x_res = numerator / denominator

    data = []
    number_of_iteration = 1
    data.append([x_res, abs(x_res - x)])

    while abs(x_res - x) > eps:
        x = x_res
        numerator = x * count_phi_for_simple_iteration(count_phi_for_simple_iteration(x)) - \
            count_phi_for_simple_iteration(x) ** 2
        denominator = count_phi_for_simple_iteration(count_phi_for_simple_iteration(x)) - \
            2 * count_phi_for_simple_iteration(x) + x
        x_res = numerator / denominator

        number_of_iteration += 1
        data.append([x_res, abs(x_res - x)])

    return data, np.arange(1, number_of_iteration + 1)


def print_plot():
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    fig.suptitle("Лабораторная работа №1")
    line1, = ax.plot(np.arange(0, 2.1, 0.1), [4 ** x for x in np.arange(0, 2.1, 0.1)])
    line2, = ax.plot(np.arange(0, 2.1, 0.1), [5 * x + 2 for x in np.arange(0, 2.1, 0.1)])

    ax.set_xlabel("ox")
    ax.set_ylabel("oy")
    ax.legend((line1, line2), [r'$f(x) = 4^x$', r'$f(x) = 5 \cdot x + 2$'])
    ax.annotate("Решение", xy=(1.69, 10.5), xytext=(1, 15), arrowprops={'facecolor': 'blue', 'shrink': 0.1})

    plt.grid()
    plt.show()


def print_data_frame(data, index, name):
    print(pd.DataFrame(data, index=index, columns=name))


if __name__ == '__main__':
    print_plot()

    names_simple_iteration = ['ak', 'bk', 'f(ak)', 'f(bk)', '(ak + bk) / 2', 'bk - ak']
    names = ['xk', '|xk - x(k - 1)|']

    e = 10 ** (-1)
    e_for_method = 10 ** (-6)
    x0, x1 = 1, 2
    pair_x01, information, order = method_of_half_division(x0, x1, e)
    print_data_frame(information, order, names_simple_iteration)
    information, order = method_of_simple_iteration((pair_x01[0] + pair_x01[1]) / 2, e_for_method)
    print_data_frame(information, order, names)
    information, order = newton_method((pair_x01[0] + pair_x01[1]) / 2, e_for_method)
    print_data_frame(information, order, names)
    information, order = steffensen_method((pair_x01[0] + pair_x01[1]) / 2, e_for_method)
    print_data_frame(information, order, names)
