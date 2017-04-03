# coding: utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import h5py
import pandas as pd
import os
from os.path import join as pjoin

from ..utils.settings import training_directory, representations_directory
from .convolution_part import convolutional_part
from ..utils.utils import *
from ..utils.callback import callback as callback_class


class first_head():

	
	def __init__(self, input_shape=(None, 2048), learning_rate=1e-4, callback=True, 
				 callback_path="./"):

		self.input = tf.placeholder(tf.float32, shape=input_shape)

		self.conversion_dict = create_conversion_model1()

		self.keep_prob = tf.placeholder(tf.float32)
		self.dropout = tf.nn.dropout(self.input, self.keep_prob)

		self.W_o = np.array([weight_variable((input_shape[1], 37)) for _ in range(23)])
		self.b_o = np.array([bias_variable([37]) for _ in range(23)])

		self.output = np.array([tf.nn.softmax(tf.matmul(self.dropout, self.W_o[k]) + self.b_o[k], 
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
			feed_dict = {self.input: x, self.keep_prob: 1}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.input: x, self.keep_prob: 1}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return predicted

	def predict(self, x, sess, all_k=True, onek=0):
		if not all_k:
			feed_dict = {self.input: x, self.keep_prob: 1}
			return sess.run(self.output[onek], feed_dict=feed_dict)
		else:
			predicted = np.zeros((x.shape[0], 23, 37))
			for k in range(23):
				feed_dict = {self.input: x, self.keep_prob: 1}
				predicted[:, k, :] = sess.run(self.output[k], feed_dict=feed_dict)
			return np.argmax(predicted, axis=-1)


	def create_train(self, learning_rate=0.001):
		self.train_steps = []
		self.true_probas = []
		for k in range(23):
			self.true_probas.append(tf.reduce_sum(- tf.log(tf.reduce_sum(tf.multiply(self.output[k], self.target[k]), axis=1))))
			self.train_steps.append(tf.train.AdamOptimizer(learning_rate).minimize(self.true_probas[k]))
		

	def f_train_step(self, x, target, sess):
		training_target = one_hot(process_target_model_1(target, self.conversion_dict))
		loss_score = 0
		for k in range(23):
			feed_dict = {self.input: x, self.keep_prob: .8, self.target[k]: training_target[:, k, :]}
			loss_score += (sess.run(tf.reduce_sum(- tf.log(tf.reduce_sum(tf.multiply(self.output[k], self.target[k]), axis=1))), feed_dict=feed_dict))
			sess.run(self.train_steps[k], feed_dict=feed_dict)
		return loss_score

	def load_weights(self, weights_path, sess):
		saver = tf.train.Saver()
		saver.restore(sess, weights_path)
		print("Model Loaded.")

	def train(self, train_representations_files, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model1_resnet.ckpt", save_path="./model1_resnet.ckpt", 
			  test_representations_files=None):

		word_file = pjoin(training_directory, "word.csv")
		word = pd.read_csv(word_file, sep=';', index_col=0)
		word['batch_nb'] = word['file'].apply(lambda x: int(x.split('/')[1]))

		#print( "%s training pictures"%x.shape[0])
		#print( "%s testing pictures"%test_x.shape[0])

		#print("Goal on the training set: %s"%np.mean(target == 36))
		#print("Goal on the testing set: %s"%np.mean(test_target == 36))

		#prediction = self.predict(x, sess)
		#print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		saver = tf.train.Saver()

		if warmstart:
			saver.restore(sess, weights_path)
			print("Model Loaded.")
		
		for i in range(1, nb_epoch + 1):

			loss = 0
			for batch_nb_m_1, representation_file in enumerate(train_representations_files):
				# Load pre-calculated representations
				h5f = h5py.File(pjoin(representations_directory, representation_file),'r')
				X = h5f['img_emb'][:]
				h5f.close()

				y = word[word['batch_nb']==batch_nb_m_1 + 1]['tag'].values

				loss += self.f_train_step(X, y, sess)

			print("Loss: %s"%loss)
		
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 1 == 0:
				if self.callback is not None:
					self.callback.store_loss(loss)
				training_accuracy = self.compute_accuracy(train_representations_files, sess, word)
				print("Training accuracy: %s" %training_accuracy)
				if self.callback is not None:
					self.callback.store_accuracy_train(training_accuracy)
				if test_representations_files is not None:
					current_accuracy = self.compute_accuracy(test_representations_files, sess, word)
					print("Validation accuracy: %s" %current_accuracy)
					if self.callback is not None:
						self.callback.store_accuracy_test(current_accuracy)
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)

		if self.callback is not None:
			self.callback.save_all(self.callback_path)

	def compute_accuracy(self, representation_files, sess, word):
		first_step = True
		for representation_file in representation_files:

				batch_nb_m_1 = int(representation_file.split(".")[0].split("_")[-1])
				
				h5f = h5py.File(pjoin(representations_directory, representation_file),'r')
				if first_step:
					X = h5f['img_emb'][:]
				else:
					X = np.vstack((X, h5f['img_emb'][:]))
				h5f.close()

				if first_step:
					y = list(word[word['batch_nb']==batch_nb_m_1]['tag'].values)
					first_step = False
				else:
					y += list(word[word['batch_nb']==batch_nb_m_1]['tag'].values)

		predicted = self.predict(X, sess)
		target = process_target_model_1(y, self.conversion_dict)
		return (np.mean(predicted == target))
		
