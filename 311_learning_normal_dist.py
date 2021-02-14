import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from samplecalls import *

learning_rate = 1.5
epochs = 5000


def model(x_value, mu_value, sigma_value):
    return tf.exp(tf.divide(tf.negative(tf.pow(tf.subtract(x_value,  mu_value), 2.0)),
                            tf.multiply(2.0, tf.pow(sigma_value, 2.0))))


def normal_dist(x, mu, sigma):
    return np.exp(- np.power((x - mu), 2.0) / np.power(sigma, 2.0))

with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.float32)
    Y = tf.compat.v1.placeholder(tf.float32)
    mu = tf.Variable(1.0, name="mu")
    sigma = tf.Variable(1.0, name="sigma")
    y_model = model(X, mu, sigma)
    cost = tf.square(Y - y_model)
    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    for epoch in range(epochs):
        for (x, y) in zip(x_train, ny_train):
            #print(x, y)
            sess.run(train_op, feed_dict={X: x, Y: y})

    mu_val = sess.run(mu)
    sig_val = sess.run(sigma)
    plt.plot(x_train, normal_dist(x_train, mu_val, sig_val))
    plt.show()
    # print(mu_val)
    # print(sig_val)
    sess.close()


