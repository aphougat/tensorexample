import tensorflow as tf

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    options = tf.compat.v1.RunOptions(output_partition_graphs=True)
    metadata = tf.compat.v1.RunMetadata()
    x = tf.constant([[1.0, 2.0]])
    negMatrix = tf.negative(x)
    result = sess.run(negMatrix, options=options, run_metadata=metadata)
    print(result)
    print(metadata.partition_graphs)
