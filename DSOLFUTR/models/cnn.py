# coding: utf-8
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
import tensorflow as tf

input_shape = (197, 197, 3)
input_tensor_shape = (None, 197, 197, 3)
class resnet:

	def __init__(self):
		self.x = tf.placeholder(tf.float32, shape=input_tensor_shape, name="input")
		self.model = ResNet50(include_top=False, input_shape=input_shape, input_tensor=self.x)
		self.maxpool3 = self.model.output

	def predict(self, input):
		return self.maxpool3.predict(input)