import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from samplecalls import *

learning_rate = 1.5
epochs = 5000


def model(x, mu, sigma):
    return tf.exp(tf.divide(tf.negative(tf.pow(x - mu, 2.0)), tf.pow(sigma, 2.0)))


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
            sess.run(train_op, feed_dict={X: x, Y: y})

    mu_val = sess.run(mu)
    sig_val = sess.run(sigma)

    print(mu_val)
    print(sig_val)
