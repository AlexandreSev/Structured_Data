# coding: utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part
from ..utils.utils import  one_hot
from ..utils.tf_func import weight_variable, bias_variable
from ..utils.callback import callback as callback_class


class first_model:
	"""
	Class for the character sequence model on ICDAR dataset
	"""
	
	def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4, cnn=None, callback=True, 
				 callback_path="./", nb_units=512):
		"""
		Parameters:
			input_shape: shape of the input tensorflow
			learning_rate: learning_rate of the AdamOptimizer
			cnn: cnn to use before this model. If none, will create it's own cnn
			callback: Boolean. Store or not the loss and the accuracy during the training_accuracy
			callback_path: where to save callbacks
			nb_units: number of units in the hidden layer
		"""
		
		if cnn is None:
			self.cnn = convolutional_part(input_shape)
		else:
			self.cnn = cnn

		flatten_shape = self.cnn.output_shape
		
		self.input = tf.reshape(self.cnn.output, [-1, flatten_shape])

		self.keep_prob = tf.placeholder(tf.float32)

		self.W_h = weight_variable((flatten_shape, nb_units))
		self.b_h = bias_variable([nb_units])

		self.h = tf.nn.relu(tf.matmul(self.input, self.W_h) + self.b_h)	

		self.dropouted_h = tf.nn.dropout(self.h, 0.7)	

		self.W_o = np.array([weight_variable((nb_units, 37)) for _ in range(23)])
		self.b_o = np.array([bias_variable([37]) for _ in range(23)])

		self.output = np.array([tf.nn.softmax(tf.matmul(self.dropouted_h, self.W_o[k]) + self.b_o[k], 
								dim=-1) for k in range(23)])
		self.target = np.array([tf.placeholder(tf.float32, shape=(None, 37)) 
								for k in range(23)])

		self.create_train(learning_rate=learning_rate)

		self.max_validation_accuracy = 0

		if callback:
			self.callback = callback_class()
			self.callback_path = callback_path
		else:
			self.callback = None

	def predict_proba(self, x, sess, all_k=True, onek=0):
		"""
		Return the probabilities for a given input
		Parameters:
			x: input of the cnn
			sess: tensorflow session
			all_k: if True, return the predictions for the 23 heads, else for only one head.
			onek: integer. If all_k is False, it will return the prediction for the head onek
		"""
		if not all_k:
			feed_dict = {self.cnn.x: x, self.keep_prob: 1.}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.cnn.x: x, self.keep_prob: 1.}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return predicted

	def predict(self, x, sess, all_k=True, onek=0):
		"""
		Return the predictions for a given input
		Parameters:
			x: input of the cnn
			sess: tensorflow session
			all_k: if True, return the predictions for the 23 heads, else for only one head.
			onek: integer. If all_k is False, it will return the prediction for the head onek
		"""
		if not all_k:
			feed_dict = {self.cnn.x: x, self.keep_prob: 1.}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.cnn.x: x, self.keep_prob: 1.}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return np.argmax(predicted, axis=-1)


	def create_train(self, learning_rate=0.001):
		"""
		Create the nodes in the tf graph used in the training phase
		parameters:
			learning_rate: learning rate of the optimiser
		"""
		self.train_steps = []
		self.true_probas = []
		for k in range(23):
			self.true_probas.append(tf.reduce_sum(- tf.log(tf.reduce_sum(tf.multiply(self.output[k], self.target[k]), axis=1))))
			self.train_steps.append(tf.train.AdamOptimizer(learning_rate).minimize(self.true_probas[k]))
		

	def f_train_step(self, x, target, sess):
		"""
		Update the weights one time for each observation
		Parameters:
			x: input of the cnn
			target: label of the input
			sess: tensorflow session
		"""
		loss_score = 0
		for k in range(23):
			feed_dict = {self.cnn.x: x, self.target[k]: target[:, k, :], self.keep_prob:1.}
			loss_score += (sess.run(self.true_probas[k], feed_dict=feed_dict))
			sess.run(self.train_steps[k], feed_dict=feed_dict)
		print("Loss: %s"%loss_score)
		return loss_score

	def load_weights(self, weights_path, sess):
		"""
		Load weights from a previous session
		Parameters:
			weights_path: path where is the file ckpt
			sess: tensorflow session
		"""
		saver = tf.train.Saver()
		saver.restore(sess, weights_path)
		print("Model Loaded.")

	def train(self, x, target, sess, nb_epoch=100, warmstart=False, 
			  weights_path="./model1.ckpt", save_path="./model1.ckpt", test_x=None, 
			  test_target=None):
		"""
		Compute the training phase
		Parameters:
			x: training sample
			target: training target
			sess: tensorflow session
			nb_epoch: number of epochs
			warmstart: if True, the model will load weights before the training_accuracy
			weights_path: where to find the ckpt file
			save_path: where to save the new weights
			text_x: testing sample. If none, there will be no validation set
			test_target: testing target
		"""

		training_target = one_hot(target)

		print( "%s training pictures"%x.shape[0])
		print( "%s testing pictures"%test_x.shape[0])

		print("Goal on the training set: %s"%np.mean(target == 36))
		print("Goal on the testing set: %s"%np.mean(test_target == 36))

		prediction = self.predict(x, sess)
		print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		saver = tf.train.Saver()

		if warmstart:
			saver.restore(sess, weights_path)
			print("Model Loaded.")
		
		for i in range(1, nb_epoch + 1):

			loss = self.f_train_step(x, training_target, sess)
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 5 == 0:
				
				if self.callback is not None:
					self.callback.store_loss(loss)
				training_accuracy = self.compute_accuracy(x, target, sess)
				print("Training accuracy: %s" %training_accuracy)
				
				if self.callback is not None:
					self.callback.store_accuracy_train(training_accuracy)
				
				if test_x is not None:
					current_accuracy = self.compute_accuracy(test_x, test_target, sess)
					print("Validation accuracy: %s" %current_accuracy)
					
					if self.callback is not None:
						self.callback.store_accuracy_test(current_accuracy)
					
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)

		if self.callback is not None:
			self.callback.save_all(self.callback_path)

	def compute_accuracy(self, x, target, sess):
		"""
		Compute the accuracy on a given set
		Parameters:
			x: input sample
			target: labels of the input
			sess: tensorflow session
		"""
		predicted = self.predict(x, sess)
		return (np.mean(predicted == target))
