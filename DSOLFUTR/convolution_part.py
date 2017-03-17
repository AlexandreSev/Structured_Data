# coding: utf-8
from settings import training_directory
from os.path import join as pjoin
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", index_col=0)

print(data.head())


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#Input

class convolutional_part():

	def __init__(self, input_shape=(None, 32, 32, 3)):
		self.x = tf.placeholder(tf.float32, shape=input_shape, name="input")

		self.W_conv1 = weight_variable([7, 7, 3, 64])
		self.b_conv1 = bias_variable([64])

		self.conv_1 = conv2d(self.x, self.W_conv1) + self.b_conv1
		self.maxpool1 = max_pool_2x2(self.conv_1)


		self.W_conv2 = weight_variable([5, 5, 64, 128])
		self.b_conv2 = bias_variable([128])

		self.conv_2 = conv2d(self.maxpool1, self.W_conv2) + self.b_conv2
		self.maxpool2 = max_pool_2x2(self.conv_2)


		self.W_conv3 = weight_variable([3, 3, 128, 256])
		self.b_conv3 = bias_variable([256])

		self.conv_3 = conv2d(self.maxpool2, self.W_conv3) + self.b_conv3
		self.maxpool3 = max_pool_2x2(self.conv_3)

	def predict(self, input):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			feed_dict = {self.x: input}
			return sess.run(self.maxpool3, feed_dict=feed_dict)

			
if __name__ == "__main__":
	test = np.load(pjoin(training_directory, "test.npy")).reshape(1, 32, 32, 3)
	model = convolutional_part()
	print(model.predict(test))
	



