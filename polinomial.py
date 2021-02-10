import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epoch = 40

trX = np.linspace(-1, 1, 101)
num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY: float = 0

for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)

trY += np.random.randn(*trX.shape) * 1.5

def model(value, weight):
    terms = []
    for index in range(num_coeffs):
        term = tf.multiply(weight[index], tf.pow(value, index))
    terms.append(term)
    return tf.add_n(terms)


with tf.compat.v1.Session() as sess:
    X = tf.compat.v1.placeholder(tf.float32)
    Y = tf.compat.v1.placeholder(tf.float32)
    w = tf.Variable([0.0] * num_coeffs, name="parameters")
    y_model = model(X, w)
    cost = tf.pow(Y - y_model, 2)
    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    for epoch in range(training_epoch):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)
    print(w_val)

plt.scatter(trX, trY)
trY2 = 0
for i in range(num_coeffs):
    trY2 += w_val[i] * np.power(trX, i)
plt.plot(trX, trY2, 'r')
plt.show()
