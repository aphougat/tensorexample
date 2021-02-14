import tensorflow as tf
import functools as ft
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
    p_y_value = normal_dist(x_train, mu_val, sig_val)
    plt.plot(x_train, p_y_value)
    plt.show()
    print("predicted value is %s" % p_y_value[33])
    print("actual value for week 35 is %s" % ny_train[33])

    error = np.power(np.power(ny_train - p_y_value, 2.0), 0.5)
    plt.bar(x_train, error)
    plt.show()

    avg_error = ft.reduce(lambda a, b: a+b, ny_train - p_y_value)
    avg_error = np.abs(avg_error) / len(x_train)
    print("average error is %s " % avg_error)
    accuracy = 1.0 - (avg_error / maxY)
    print("accuracy is %s" % accuracy)

    sess.close()


