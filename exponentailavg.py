import tensorflow as tf
import numpy as np
g_1 = tf.Graph()
with tf.compat.v1.Session(graph=g_1) as sess:
    raw_data = np.random.normal(10, 1, 100)
    alpha = tf.constant(0.05)
    curr_value = tf.compat.v1.placeholder(tf.float32)
    prev_avg = tf.Variable(0.0, name="curr_value")
    update_avg = alpha * curr_value + (1 - alpha) * prev_avg
    avg_hist = tf.compat.v1.summary.scalar("running_avg", update_avg)
    value_hist = tf.compat.v1.summary.scalar("incoming_values", curr_value)
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("./logs")
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    #sess.add_graph(sess.graph)
    for i in range(len(raw_data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        #curr_avg = sess.run(alpha * raw_data[i] + (1 - alpha) * prev_avg)
        sess.run(tf.compat.v1.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)
        writer.add_summary(summary_str, i)
