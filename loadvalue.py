import tensorflow as tf

with tf.compat.v1.Session() as sess:
    spikes = tf.Variable([False] * 8, name="spikes")
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, "./spikes.ckpt")
    print(spikes.eval())
