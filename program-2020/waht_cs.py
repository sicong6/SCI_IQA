# -*- coding:UTF-8 -*-

import tensorflow as tf


a = tf.constant([[5], [4], [6], [5], [6], [5], [5], [4], [6], [5], [5], [5], [4], [6], [5], [5], ])
b = tf.constant(5)
b = tf.reshape(b, [-1, 1])
loss = tf.losses.mean_squared_error(b, a)

with tf.Session(graph=graph) as sess:
	sd,df,fg = sess.run([a,b,loss])
	print(a.shape)
	print(b)
	print(loss)




