# coding: utf-8

from os.path import join as pjoin
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ..utils.settings import training_directory
from ..utils.tf_func import *


class convolutional_part:
	"""
	Homemade CNN, three convolutions and three maxpoolings
	"""

	def __init__(self, input_shape=(None, 32, 32, 1)):
		self.x = tf.placeholder(tf.float32, shape=input_shape, name="input")

		self.W_conv1 = weight_variable([7, 7, 1, 64])
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
		self.output = max_pool_2x2(self.conv_3)

		self.output_shape =  input_shape[1] // 8 * input_shape[2] // 8 * 256

	def predict(self, x, sess):
		"""
		Predict the output of the cnn 
		parameters:
			x: input of the cnn
			sess: tensorflow session
		"""
		feed_dict = {self.x: x}
		return sess.run(self.output, feed_dict=feed_dict)