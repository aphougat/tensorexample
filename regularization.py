import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

learning_rate = 0.001
training_epochs = 1000

x_dataset = np.linspace(-1, 1, 1000)
num_coff = 9
reg_lambda = 0.
y_dataset_params = [0.] * num_coff
y_dataset_params[2] = 1.0
y_dataset = 0

for i in range(num_coff):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

(x_train, x_test, y_train, y_test) = train_test_split(x_dataset, y_dataset, train_size=0.7)

def model(xvalue, weight):
    terms = []
    for i in range(num_coff):
        term = tf.multiply(weight[i], tf.pow(xvalue, i))
        terms.append(term)
    return tf.add_n(terms)


with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.float32)
    Y = tf.compat.v1.placeholder(tf.float32)
    w = tf.Variable([0.] * num_coff, name="parameters")
    y_model = model(X, w)
    cost = tf.compat.v1.div(tf.add(tf.reduce_sum(tf.square(Y - y_model)),
                                   tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))), 2 * x_train.size)
    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    for reg_lambda in np.linspace(0, 1, 100):
        for epoch in range(training_epochs):
            sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        final_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
        print("reg_lambda", reg_lambda)
        print("final_cost", final_cost)

