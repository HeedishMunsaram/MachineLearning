# x(age) = 10  20  30  40  50  60  70  80  90  100
# y(population have mob) = 40   80  90  81  65  50  35  30  15  5
# draw scatter diagram
# calculate gradient & Y-intercept
# plot the new regression line

import numpy as np
import matplotlib.pyplot as plt

def calculate_coef(a, p):

    size = np.size(a)

    mean_a = np.mean(a)
    mean_p = np.mean(p)

    crossdeviation_ap = np.sum(p*a) - size*mean_a*mean_p
    crossdeviation_aa = np.sum(a*a) - size*mean_a*mean_a

    b1 = crossdeviation_ap/crossdeviation_aa
    b2 = mean_p - b1 * mean_a

    return b2, b1


def function_to_plot_regression(a, p, w):

    plt.scatter(a, p, color="b", marker="*", s=30)

    p_predict = w[0] + w[1] * a

    plt.plot(a, p_predict, color="y")

    plt.xlabel('a')
    plt.ylabel('p')

    plt.show()


def main():

    a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    p = np.array([40, 80, 90, 81, 65, 50, 35, 30, 15, 5])

    w = calculate_coef(a, p)
    print("estimated coefficients:\nb_0={} \nb_1 = {}".format(w[0], w[1]))

    function_to_plot_regression(a, p, w)


if __name__ == "__main__":
    main()


