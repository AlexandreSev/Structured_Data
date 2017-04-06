# coding: utf-8
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
import tensorflow as tf

class resnet:
	"""
	CNN resnet
	"""

	def __init__(self, input_shape=(197, 197, 3)):
		
		input_tensor_shape = (None, input_shape[0], input_shape[1], input_shape[2])

		self.x = tf.placeholder(tf.float32, shape=input_tensor_shape, name="input")
		self.model = ResNet50(include_top=False, input_shape=input_shape, input_tensor=self.x)
		self.output = self.model.output

	def predict(self, x):
		"""
		Compute the output of resnet for the input
		"""
		return self.output.predict(x)
		