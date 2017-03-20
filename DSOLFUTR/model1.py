# input shape None, 4, 4, 256
# compte tf.matmul # avec en gros le x comme le poids pour avoir 
# y = tf.matmul(X, W)

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
return tf.Variable(initial)



class first_model():
	"""
	"""

	def __init__(self, input_shape=(None, 4, 4, 256)):
		self.W = weight_variable((4096, 37, 23))
		self.b = bias_variable((37,23))
		self.input = tf.placeholder(tf.float32, shape=input_shape)
		self.input = tf.reshape(self.input, (-1, 4096))
		self.output = tf.nn.softmax(tf.matmul(self.input, self.W) + self.b, dim=0)
	

	def predict(self, input):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.input: input}
			return sess.run(self.output, feed_dict=feed_dict)

	def create_one_layer(self, position, input):
		"""
		"""
		with 
		tf.nn.softmax(logits, dim=-1, name=None) 
		return 
