import tensorflow as tf

#tf.compat.v1.enable_eager_execution()
sess = tf.compat.v1.InteractiveSession(grapg=c)
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
devices = sess.list_devices()
for d in devices:
  print(d.name)
# We can just use 'c.eval()' without passing 'sess'
print(c)
print(sess.run(c))
print(c.eval())
sess.close()
