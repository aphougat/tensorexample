import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normal_dist(x, mu, sigma):
    return np.exp(- np.power((x - mu), 2.0) / np.power(sigma, 2.0))


x_values = np.linspace(-3, 3, 120)
for (mu, sig) in [(-1, 1), (0, 2), (2, 3)]:
    plt.plot(x_values, normal_dist(x_values, mu, sig))

plt.show()
