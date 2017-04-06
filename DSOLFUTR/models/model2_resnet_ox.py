#coding:utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle
import pandas as pd
import os
from os.path import join as pjoin
import h5py

from ..utils.ngrams import get_ngrams, get_dict_ngrams
from ..utils.utils import process_target_model_2
from ..utils.tf_func import weight_variable, bias_variable
from ..utils.settings import training_directory, representations_directory, ox_directory
from ..utils.callback import callback as callback_class

class second_head():
	"""
	Class for the character sequence model if representations of MJSynth have already be computed
	"""

	def __init__(self, input_shape=(None, 2048), learning_rate=1e-4, callback=True, 
				 callback_path="./"):
		"""
		Parameters:
			input_shape: shape of the input tensorflow
			learning_rate: learning_rate of the AdamOptimizer
			callback: Boolean. Store or not the loss and the accuracy during the training_accuracy
			callback_path: where to save callbacks
		"""
		
		self.ngrams = get_ngrams()

		self.dict_n_grams = get_dict_ngrams(self.ngrams)

		self.output_size = len(self.ngrams)
		
		self.input = tf.placeholder(tf.float32, shape=input_shape)

		self.keep_prob = tf.placeholder(tf.float32)
		self.dropout = tf.nn.dropout(self.input, self.keep_prob)

		self.W_o = weight_variable(shape=(input_shape[1], self.output_size))
		self.b_o = bias_variable(shape=[self.output_size])

		self.output = tf.sigmoid(tf.matmul(self.dropout, self.W_o) + self.b_o) 

		self.target = tf.placeholder(tf.float32, shape=(None, self.output_size)) 

		self.create_train(learning_rate=learning_rate)

		self.max_validation_accuracy = 0

		if callback:
			self.callback = callback_class()
			self.callback_path = callback_path
		else:
			self.callback = None
	
	def predict_proba(self, x, sess):
		"""
		Return the probabilities for a given input
		Parameters:
			x: input of the cnn
			sess: tensorflow session
		"""
		feed_dict = {self.input: x, self.keep_prob: 1}
		return sess.run(self.output, feed_dict=feed_dict)

	def predict(self, x, sess, treshold=0.5):
		"""
		Return the predictions for a given input
		Parameters:
			x: input of the cnn
			sess: tensorflow session
			treshold: treshold between 0 and 1.
		"""	
		feed_dict = {self.input: x, self.keep_prob: 1}
		predicted = sess.run(self.output, feed_dict=feed_dict)
		return (predicted > treshold).astype(np.int)

	def create_train(self, learning_rate=0.001):
		"""
		Create the nodes in the tf graph used in the training phase
		parameters:
			learning_rate: learning rate of the optimiser
		"""
		self.loss = tf.nn.l2_loss(self.output - self.target)
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		

	def f_train_step(self, x, target, sess):
		"""
		Update the weights one time for each observation
		Parameters:
			x: input of the cnn
			target: label of the input
			sess: tensorflow session
		"""
		training_target = process_target_model_2(target, self.dict_n_grams)
		feed_dict = {self.input: x, self.keep_prob: .8, self.target: training_target}
		loss_score = (sess.run(self.loss, feed_dict=feed_dict))
		sess.run(self.train_step, feed_dict=feed_dict)
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

	def train(self, train_representations_files, sess, nb_epoch=100, warmstart=False, 
			  weights_path="./model2_resnet_ox.ckpt", save_path="./model2_resnet_ox.ckpt", 
			  test_representations_files=None):
		"""
		Compute the training phase
		Parameters:
			train_representations_files: files used for training
			sess: tensorflow session
			nb_epoch: number of epochs
			warmstart: if True, the model will load weights before the training_accuracy
			weights_path: where to find the ckpt file
			save_path: where to save the new weights
			text_representations_files: files used for testing
		"""

		saver = tf.train.Saver()

		for i in range(1, nb_epoch + 1):

			loss = 0
			for batch_nb_m in train_representations_files:
				batch_nb_m_1 = str(batch_nb_m)
				if os.path.isfile(pjoin(ox_directory, "representations", "img_emb_"+ batch_nb_m_1 + ".h5")):
					
					# Load pre-calculated representations
					h5f = h5py.File(pjoin(ox_directory, "representations", "img_emb_"+ batch_nb_m_1 + ".h5"),'r')
					X = h5f['img_emb'][:]
					h5f.close()

					y = pickle.load(open(pjoin(ox_directory, "targets", "target_" + batch_nb_m_1 + ".txt"), "rb"))

					temp = self.f_train_step(X, y, sess)

					print("Loss of batch %s: %s"%(batch_nb_m, temp))
					loss += temp

			print("Loss: %s"%loss)

			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 1 == 0:
				if self.callback is not None:
					self.callback.store_loss(loss)
				training_accuracy = self.compute_accuracy(train_representations_files, sess)
				print("Training accuracy: %s" %training_accuracy)
				if self.callback is not None:
					self.callback.store_accuracy_train(training_accuracy)
				if test_representations_files is not None:
					current_accuracy = self.compute_accuracy(test_representations_files, sess)
					print(current_accuracy)
					print("Validation accuracy: %s" %current_accuracy)
					if self.callback is not None:
						self.callback.store_accuracy_test(current_accuracy)
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)
		
		if self.callback is not None:
		 	self.callback.save_all(self.callback_path)


	def compute_accuracy(self, representation_files, sess):
		"""
		Compute the accuracy on a given set
		Parameters:
			representation_files: files used in input
			sess: tensorflow session
		"""
		nb_true = 0
		nb_total = 0
		for batch_nb_m in train_representations_files:
				batch_nb_m_1 = str(batch_nb_m)
				if os.path.isfile(pjoin(ox_directory, "representations", "img_emb_"+ batch_nb_m_1 + ".h5")):
					
					# Load pre-calculated representations
					h5f = h5py.File(pjoin(ox_directory, "representations", "img_emb_"+ batch_nb_m_1 + ".h5"),'r')
					
					X = h5f['img_emb'][:]
					h5f.close()

					y = pickle.load(open(pjoin(ox_directory, "targets", "target_" + batch_nb_m_1 + ".txt"), "rb"))

					
					predicted = self.predict(X, sess)
					target = process_target_model_2(y, self.dict_n_grams)

					nb_true += np.sum(predicted == target)
					nb_total += len(predicted)
		return nb_true / nb_total
