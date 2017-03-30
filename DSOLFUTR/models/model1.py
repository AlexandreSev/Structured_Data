# coding: utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part
from ..utils.utils import weight_variable, bias_variable, one_hot
from ..utils.callback import callback as callback_class


class first_model():

	
	def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4, cnn=None, callback=True, 
				 callback_path="./"):
		
		if cnn is None:
			self.cnn = convolutional_part(input_shape)
		else:
			self.cnn = cnn

		self.input = tf.reshape(self.cnn.maxpool3, [-1, 4096])

		self.W_h = weight_variable((4096, 512))
		self.b_h = bias_variable([512])

		self.h = tf.nn.relu(tf.matmul(self.input, self.W_h) + self.b_h)	

		self.dropouted_h = tf.nn.dropout(self.h, 0.7)	

		self.W_o = np.array([weight_variable((512, 37)) for _ in range(23)])
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
		if not all_k:
			feed_dict = {self.cnn.x: x}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.cnn.x: x}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return predicted

	def predict(self, x, sess, all_k=True, onek=0):
		if not all_k:
			feed_dict = {self.cnn.x: x}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.cnn.x: x}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return np.argmax(predicted, axis=-1)


	def create_train(self, learning_rate=0.001):
		self.train_steps = []
		self.true_probas = []
		for k in range(23):
			self.true_probas.append(tf.reduce_sum(- tf.log(tf.reduce_sum(tf.multiply(self.output[k], self.target[k]), axis=1))))
			self.train_steps.append(tf.train.AdamOptimizer(learning_rate).minimize(self.true_probas[k]))
		

	def f_train_step(self, x, target, sess):
		loss_score = 0
		for k in range(23):
			feed_dict = {self.cnn.x: x, self.target[k]: target[:, k, :]}
			loss_score += (sess.run(tf.reduce_sum(- tf.log(tf.reduce_sum(tf.multiply(self.output[k], self.target[k]), axis=1))), feed_dict=feed_dict))
			sess.run(self.train_steps[k], feed_dict=feed_dict)
		print("Loss: %s"%loss_score)
		return loss_score

	def load_weights(self, weights_path, sess):
		saver = tf.train.Saver()
		saver.restore(sess, weights_path)
		print("Model Loaded.")

	def train(self, x, target, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model1.ckpt", save_path="./model1.ckpt", test_x=None, 
			  test_target=None):

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
		predicted = self.predict(x, sess)
		return (np.mean(predicted == target))
		
