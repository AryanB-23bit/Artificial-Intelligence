import math
import random


def main():
    init_w0 = 0.25
    init_w1 = 0.25
    learning_rate = 0.001
    max_iter = 1000000
    x_vals = [2, 4, 6, 7, 8, 10]
    y_vals = [5, 7, 14, 14, 17, 19]

    w0BGD, w1BGD, epochBGD = run_BGD(init_w0, init_w1, x_vals, y_vals, learning_rate, max_iter)
    print("Batch Gradient Descent Line Equation: y = " + str(w0BGD) + " + " + str(w1BGD) + "x")
    print("Batch Gradient Descent Epoch: " + str(epochBGD))
    print("Batch Gradient Descent f(x = 5) = ", w0BGD + (w1BGD * 5))
    print("Batch Gradient Descent f(x = -100) = ", w0BGD + (w1BGD * -100))
    print("Batch Gradient Descent f(x = 100) = ", w0BGD + (w1BGD * 100))

    print("\n" * 2)

    w0SGD, w1SGD, epochSGD = run_SGD(init_w0, init_w1, x_vals, y_vals, learning_rate, max_iter)
    print("Stochastic Gradient Descent Line Equation: y = " + str(w0SGD) + " + " + str(w1SGD) + "x")
    print("Stochastic Gradient Descent Epoch: " + str(epochSGD))
    print("Stochastic Gradient Descent f(x = 5) = ", w0SGD + (w1SGD * 5))
    print("Stochastic Gradient Descent f(x = -100) = ", w0SGD + (w1SGD * -100))
    print("Stochastic Gradient Descent f(x = 100) = ", w0SGD + (w1SGD * 100))


def BGD(n_w0, n_w1, x_vals, y_vals, learning_rate):
    w0 = 0
    w1 = 0
    for i in range(len(x_vals)):
        w0 += y_vals[i] - (n_w0 + n_w1 * x_vals[i])
        w1 += (y_vals[i] - (n_w0 + n_w1 * x_vals[i])) * x_vals[i]

    return n_w0 + learning_rate * w0, n_w1 + learning_rate * w1


def run_BGD(w0, w1, x_vals, y_vals, learning_rate, max_iter):
    iteration = 0
    new_w0, new_w1 = BGD(w0, w1, x_vals, y_vals, learning_rate)
    while (abs(new_w0 - w0) > math.pow(10, -10)) and iteration <= max_iter:
        iteration += 1
        w0 = new_w0
        new_w0, new_w1 = BGD(new_w0, new_w1, x_vals, y_vals, learning_rate)
    return new_w0, new_w1, iteration


def SGD(n_w0, n_w1, x_vals, y_vals, learning_rate):
    return learning_rate * (y_vals - (n_w1 * x_vals + n_w0)) + n_w0, \
           learning_rate * ((y_vals - (n_w1 * x_vals + n_w0)) * x_vals) + n_w1


def run_SGD(w0, w1, x_vals, y_vals, learning_rate, max_iter):
    iteration = 0
    new_w0, new_w1 = SGD(w0, w1, x_vals[0], y_vals[0], learning_rate)
    while (abs(new_w0 - w0) > math.pow(10, -10)) and iteration < max_iter:
        rand_int = random.randint(0, len(x_vals) - 1)
        iteration += 1
        w0 = new_w0
        w1 = new_w1
        new_w0, new_w1 = SGD(w0, w1, x_vals[rand_int], y_vals[rand_int], learning_rate)
    return new_w0, new_w1, iteration


if __name__ == '__main__':
    main()
