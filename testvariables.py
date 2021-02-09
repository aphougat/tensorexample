import tensorflow as tf

with tf.compat.v1.Session() as sess:
    matrix = [1.0, 2.0, 5.0, 4.0, 8.0, 6.0, .2, .5, .6]
    spikes = tf.Variable([False] * len(matrix), name='spikes')
    spikes.initializer.run()
    saver = tf.compat.v1.train.Saver()
    for i in range(1, len(matrix)):
        spikes_val = spikes.eval()
        if matrix[i] - matrix[i - 1] > 2:
            #updater = tf.compat.v1.assign(spike[i], True)
            spikes_val[i] = True
            #sess.run(updater)
        else:
            spikes_val[i] = False
            #updater = tf.compat.v1.assign(spike[i], False)
            #sess.run(updater)
        updater = tf.compat.v1.assign(spikes, spikes_val)
        sess.run(updater)
    print("Spike", spikes.eval())
    save_path = saver.save(sess, "spikes.ckpt")
    print("spikes data saved in file %s" % save_path)
