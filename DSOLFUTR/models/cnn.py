# coding: utf-8

#import tensorflow as tf
#from tensorflow.contrib.keras.python.keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import ResNet50

input_shape=(197, 197, 3)

class convolutional_part():

	def __init__(self):
		#self.maxpool3 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None)
		self.maxpool3 = ResNet50(include_top=False, input_shape=input_shape)

	def predict(self, input):
		#with tf.Session() as sess:
		#	sess.run(tf.global_variables_initializer())
		#	feed_dict = {self.x: input}
		#	return sess.run(self.maxpool3, feed_dict=feed_dict)
		return self.maxpool3.predict(input)