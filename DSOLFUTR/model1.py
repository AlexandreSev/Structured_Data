# input shape None, 4, 4, 256
# compte tf.matmul # avec en gros le x comme le poids pour avoir 
# y = tf.matmul(X, W)

import tensorflow as tf
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0., shape=shape)
	return tf.Variable(initial)



class first_model():
	"""
	"""

	def __init__(self, input_shape=(None, 4, 4, 256)):
		self.W = np.array([weight_variable((4096, 37)) for _ in range(23)])
		self.b = np.array([bias_variable([37]) for _ in range(23)])
		self.input = tf.placeholder(tf.float32, shape=input_shape)
		self.reshapeinput = tf.reshape(self.input, [-1, 4096], name="blabl")
		# print(tf.shape(self.input))
		# with tf.Session() as sess:
		# 	batch_size = sess.run(tf.shape(self.input))[0]
		# self.output = np.empty((batch_size, 23, 37)) # None => dim qui peut varier, -1 c'est la dimension qui convient 
		self.output = np.array([tf.nn.softmax(tf.matmul(self.reshapeinput, self.W[k]) + self.b[k], dim=-1) for k in range(23)])
		# for k in range(23): #W.shape[2]
		# 	self.output[:,k,:] = tf.nn.softmax(tf.matmul(self.input, self.W[k]) + self.b[k], dim=0)
			#output[k,:,:] = self.input.dot(self.W[:,:,k])
			# tf.nn.softmax(tf.matmul(A, W) , dim=0)
	

	def predict(self, input):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.input: input}
			return sess.run(self.output[0], feed_dict=feed_dict)

	def create_one_layer(self, position, input):
		"""
		"""
		with tf.Session() as sess:
			tf.nn.softmax(logits, dim=-1, name=None) 
			return 

if __name__== '__main__':
	a = first_model()
	test = np.load('test.npy').reshape(1,32,32,3)
	from convolution_part import *
	model = convolutional_part()
	b = model.predict(test)
	print(a.predict(b).sum())