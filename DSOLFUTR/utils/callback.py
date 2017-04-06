# coding: utf-8
from os.path import join as pjoin
import numpy as np


class callback:
	"""
	Class used to store accuracy and loss during training
	"""
	
	def __init__(self):
		self.loss = []
		self.accuracy_train = []
		self.accuracy_test = []

	def store_loss(self, loss):
		self.loss.append(loss)

	def store_accuracy_train(self, accuracy):
		self.accuracy_train.append(accuracy)

	def store_accuracy_test(self, accuracy):
		self.accuracy_test.append(accuracy)

	def store_loss_accuracy(self, loss, train_accuracy, test_accuracy):
		self.store_loss(loss)
		self.store_accuracy_train(train_accuracy)
		self.store_accuracy_test(test_accuracy)

	def save_all(self, path, name=""):
		loss_path = pjoin(path, "loss_callback" + name+ ".npy")
		accuracy_train_path = pjoin(path, "accuracy_train_callback" + name+ ".npy")
		accuracy_test_path = pjoin(path, "accuracy_test_callback" + name+ ".npy")
		np.save(loss_path, self.loss)
		np.save(accuracy_train_path, self.accuracy_train)
		np.save(accuracy_test_path, self.accuracy_test)
