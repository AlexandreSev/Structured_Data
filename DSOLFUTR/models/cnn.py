# coding: utf-8
from keras.applications.resnet50 import ResNet50

input_shape=(197, 197, 3)

class resnet:

	def __init__(self):
		self.model = ResNet50(include_top=False, input_shape=input_shape)
		self.maxpool3 = self.model.output

	def predict(self, input):
		return self.maxpool3.predict(input)