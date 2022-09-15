import math as m
import numpy as np


def f1(x):
    return m.sin(2 * x) * m.log(x + 5, m.e)


def f2(x):
    return m.sqrt(2 * abs(x) + x ** 2)


def count_optimal_knots(n, section):  # чебышёвские узлы (формула)
    x_k = []
    for i in range(n + 1):
        coefficient = m.cos(((2 * i + 1) / (2 * n + 2)) * m.pi)
        x_k.append(sum(section) / 2 + ((section[1] - section[0]) / 2) * coefficient)

    return x_k


def count_equidistant_nodes(n, section):  # равноотстоящие узлы (формула)
    x_k = [section[0]]
    for i in range(1, n + 1):
        x_k.append(section[0] + i * (section[1] - section[0]) / n)

    return x_k


def pnx(x, numbers, knots):  # значение многочлена в точке(многочлен в форме Ньютона)
    value_of_polynomial = numbers[0]
    array = []
    for knot in knots:
        array.append(x - knot)

    for i in range(1, len(numbers)):
        temp = numbers[i]
        for j in range(i):
            temp *= array[j]
        value_of_polynomial += temp

    return value_of_polynomial


def create_x_(section):  # массив точек, я ищу значеине многочлена в этих точках,чтобы потом считать погрешность
    x_ = []  # формула для подсчёта данных точек есть в условии лабы
    for i in range(101):
        x_.append(section[0] + (i * (section[1] - section[0])) / 100)

    return x_


def create_polynomial(n, knots, fun, index_id):  # создаю многочлен в форме Ньютона
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):  # считаю значения скобочек, которые мне нужны для построения многочлена
        matrix[i][0] = fun[index_id](knots[i])

    # использую таблицу разделённых разностей для подсчёта коэффициентов многочлена
    counter = 0
    for j in range(1, n + 1):
        for k in range(n + 1 - j):
            matrix[k][j] = (matrix[k + 1][j - 1] - matrix[k][j - 1]) / (knots[k + 1 + counter] - knots[k])
        counter += 1
    return matrix[0]


def view_of_polynomial(numbers, knots):  # вывожу вид многочлена в форме Ньютона
    res = f"{numbers[0]}"
    res += " + "
    for i in range(1, len(numbers)):
        temp = f"{numbers[i]}"
        for j in range(i):
            temp += f"(x - {knots[j]})"
        res += temp
        res += " + "
    return res


def count_accuracy(section, numbers, knots, func, index_id):  # считаю погрешность, формула в условии лабы
    new_x_ = create_x_(section)
    result_r = [0.0] * 101
    for k in range(101):
        result_r[k] = abs(pnx(new_x_[k], numbers, knots) - func[index_id](new_x_[k]))

    return max(result_r)


if __name__ == '__main__':
    a_b = np.array([-2, 2])  # мой отрезок
    functions = [f1, f2]  # массив в функций, я по индексу обращаюсь к одной из функций, чтобы одну функцию импользовать
    # для нескольки разных функций

    for index in range(5, 21, 5):  # Первая функция чебышёвские узлы
        print(f"Polynomial order for f1 {index} with optimal knots")
        coefficients1_1 = create_polynomial(index, count_optimal_knots(index, a_b), functions, 0)
        result1_1 = view_of_polynomial(coefficients1_1, count_optimal_knots(index, a_b))
        print(result1_1)
        print(count_accuracy(a_b, coefficients1_1, count_optimal_knots(index, a_b), functions, 0))
        print()

    for index in range(5, 21, 5):  # Первая функция равноотстоящие узлы
        print(f"Polynomial order for f1 {index} with equal distance knots")
        coefficients1_2 = create_polynomial(index, count_equidistant_nodes(index, a_b), functions, 0)
        result1_2 = view_of_polynomial(coefficients1_2, count_equidistant_nodes(index, a_b))
        print(result1_2)
        print(count_accuracy(a_b, coefficients1_2, count_equidistant_nodes(index, a_b), functions, 0))
        print()

    for index in range(5, 21, 5):  # Вторая функция чебышёвские узлы
        print(f"Polynomial order for f2 {index} with optimal knots")
        coefficients2_1 = create_polynomial(index, count_optimal_knots(index, a_b), functions, 1)
        result2_1 = view_of_polynomial(coefficients2_1, count_optimal_knots(index, a_b))
        print(result2_1)
        print(count_accuracy(a_b, coefficients2_1, count_optimal_knots(index, a_b), functions, 1))
        print()

    for index in range(5, 21, 5):  # Вторая функция равноотстоящие узлы
        print(f"Polynomial order for f2 {index} with equal distance knots")
        coefficients2_2 = create_polynomial(index, count_equidistant_nodes(index, a_b), functions, 1)
        result2_2 = view_of_polynomial(coefficients2_2, count_equidistant_nodes(index, a_b))
        print(result2_2)
        print(count_accuracy(a_b, coefficients2_2, count_equidistant_nodes(index, a_b), functions, 1))
        print()
