# input shape None, 4, 4, 256
# compte tf.matmul # avec en gros le x comme le poids pour avoir 
# y = tf.matmul(X, W)

import tensorflow as tf
import numpy as np

from .convolution_part import convolutional_part

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0., shape=shape)
	return tf.Variable(initial)



class first_model():
	"""
	"""

	def __init__(self, input_shape=(None, 32, 32, 3)):
		
		self.cnn = convolutional_part(input_shape)

		self.W = np.array([weight_variable((4096, 37)) for _ in range(23)])
		self.b = np.array([bias_variable([37]) for _ in range(23)])
		self.input = tf.reshape(self.cnn.maxpool3, [-1, 4096])
		self.output = np.array([tf.nn.softmax(tf.matmul(self.input, self.W[k]) + self.b[k], dim=-1) 
								for k in range(23)])
		self.target = np.array([tf.placeholder(tf.float32, shape=(None, 37)) 
								for k in range(23)])

		self.create_train()
	

	def predict(self, x, all_k=True, onek=0):
		if not all_k:
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				feed_dict = {self.cnn.x: x}
				return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			with tf.Session() as sess:
				for k in range(23):
					sess.run(tf.global_variables_initializer())
					feed_dict = {self.cnn.x: x}
					predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return np.argmax(predicted, axis=-1)

	def create_train(self, learning_rate=0.001):
		self.train_steps = []
		self.true_probas = []
		for k in range(23):
			self.true_probas.append(tf.reduce_sum(- tf.multiply(self.output[k], self.target[k])))
			self.train_steps.append(tf.train.AdamOptimizer(learning_rate).minimize(self.true_probas[k]))
		

	def train_step(self, x, target, sess):
			for k in range(23):
				feed_dict = {self.cnn.x: x, self.target[k]: target[:, k, :]}
				sess.run(self.train_steps[k], feed_dict=feed_dict)

if __name__== '__main__':
	a = first_model()
	test = np.load('test.npy').reshape(1,32,32,3)