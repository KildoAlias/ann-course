import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def generateTimeseries():
    x = 1.5
    x_next = []

    for i in range(25):
        x_next.append(x)
        x *= 0.9

    for i in range(1475):
        x_next.append(x_next[-1] + (0.2*x_next[-25]) /
                      (1+x_next[-25]**10) - 0.1 * x_next[-1])

    return x_next


if __name__ == "__main__":

    plot_t = np.linspace(1, 1500, 1500)

    x = generateTimeseries()
    x = np.array(x)

    # Plot the time series
    plt.plot(plot_t, x)
    plt.show()
    # plt.pause(1)

    # print(x.shape)
    # x_train = x[0:600]
    # x_val = x[600:1000]
    # x_test = x[1000:]

    # print(x_train.shape)
    # print(x_val.shape)
    # print(x_test.shape)

    t_vec = np.linspace(301, 1499, 1199)

    print(t_vec.shape)
    x_in = np.array([[x[300-20]], [x[300-15]],
                     [x[300-10]], [x[300-5]], [x[300]]])

    out = np.array([x[300+5]])

    for t in t_vec:
        t = int(t)
        x_temp = np.array([[x[t-20]], [x[t-15]],
                           [x[t-10]], [x[t-5]], [x[t]]])
        x_in = np.hstack((x_in, x_temp))

        out_temp = np.array(x[t+5])
        out = np.hstack((out, out_temp))

    print("shape pf x= ", x_in.shape)
    print("sahpe of out=", out.shape)
