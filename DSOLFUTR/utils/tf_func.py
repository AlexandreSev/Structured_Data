# coding: utf-8
import tensorflow as tf

def weight_variable(shape):
	"""
	Create a tf Variable corresponding to the matrix W in NN
	parameters:
		shape: size of the variable
	"""
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	"""
	Create a tf Variable corresponding to the biais b in NN
	parameters:
		shape: size of the biais
	"""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	"""
	Convolution with standard strides and padding
	parameters:
		x: input tensor
		W: kernel tensor
	"""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""
	Max pooling with standart strides and padding
	parameters:
		x: input tensor
	"""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
