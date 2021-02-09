import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1
training_epoch = 100

x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33


def model(val, weight):
    return tf.multiply(val, weight)


with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.float32)
    Y = tf.compat.v1.placeholder(tf.float32)
    w = tf.Variable(0.0, name="weights")
    y_model = model(X, w)
    cost = tf.square(Y - y_model)

    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    for epoch in range(training_epoch):
        for (x, y) in zip(x_train, y_train):
            sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)
plt.scatter(x_train, y_train)
y_learning = x_train * w_val
plt.plot(x_train, y_learning, 'r')
plt.show()
