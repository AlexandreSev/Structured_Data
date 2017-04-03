#coding:utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle
import pandas as pd
import os
from os.path import join as pjoin
import h5py

from ..utils.settings import training_directory, representations_directory
from ..utils.ngrams import get_ngrams, get_dict_ngrams
from ..utils.utils import weight_variable, bias_variable, process_target_model_2
import h5py

from ..utils.settings import training_directory, representations_directory
from ..utils.callback import callback as callback_class

class second_head():
	"""
	"""

	def __init__(self, input_shape=(None, 2048), learning_rate=1e-4, cnn=None, callback=True, 
				 callback_path="./"):
		
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
		feed_dict = {self.input: x, self.keep_prob: 1}
		return sess.run(self.output, feed_dict=feed_dict)

	def predict(self, x, sess, treshold=0.5):
		feed_dict = {self.input: x, self.keep_prob: 1}
		predicted = sess.run(self.output, feed_dict=feed_dict)
		return (predicted > treshold).astype(np.int)

	def create_train(self, learning_rate=0.001):
		self.loss = tf.nn.l2_loss(self.output - self.target)
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		

	def f_train_step(self, x, target, sess):
		training_target = process_target_model_2(target, self.dict_n_grams)
		feed_dict = {self.input: x, self.keep_prob: .8, self.target: training_target}
		loss_score = (sess.run(self.loss, feed_dict=feed_dict))
		sess.run(self.train_step, feed_dict=feed_dict)
		return loss_score

	def load_weights(self, weights_path, sess):
		saver = tf.train.Saver()
		saver.restore(sess, weights_path)
		print("Model Loaded.")

	def train(self, train_representations_files, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model2_resnet.ckpt", save_path="./model2_resnet.ckpt", 
			  test_representations_files=None):
		
		#print( "%s training pictures"%x.shape[0])
		#print( "%s testing pictures"%test_x.shape[0])

		#print("Goal on the training set: %s"%np.mean(target == 0))
		#print("Goal on the testing set: %s"%np.mean(test_target == 0))

		#prediction = self.predict(x, sess)
		#print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		word_file = pjoin(training_directory, "word.csv")
		word = pd.read_csv(word_file, sep=';', index_col=0)
		word['batch_nb'] = word['file'].apply(lambda x: int(x.split('/')[1]))
		saver = tf.train.Saver()

		#if warmstart:
		#	saver.restore(sess, weights_path)
		#	print("Model Loaded.")

		for i in range(1, nb_epoch + 1):

			loss = 0
			for representation_file in train_representations_files:

				batch_nb_m_1 = int(representation_file.split(".")[0].split("_")[-1])
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
					print(current_accuracy)
					print("Validation accuracy: %s" %current_accuracy)
					if self.callback is not None:
						self.callback.store_accuracy_test(current_accuracy)
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)
		
		# for i in range(1, nb_epoch + 1):

		# 	loss = self.f_train_step(x, target, sess)
		# 	print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
		# 	if i % 5 == 0:
		# 		if self.callback is not None:
		# 			self.callback.store_loss(loss)
		# 		training_accuracy = self.compute_accuracy(x, target, sess)
		# 		print("Training accuracy: %s" %training_accuracy)
		# 		if self.callback is not None:
		# 			self.callback.store_accuracy_train(training_accuracy)
		# 		if test_x is not None:
		# 			current_accuracy = self.compute_accuracy(test_x, test_target, sess)
		# 			print("Validation accuracy: %s" %current_accuracy)
		# 			if self.callback is not None:
		# 				self.callback.store_accuracy_test(current_accuracy)
		# 			if current_accuracy > self.max_validation_accuracy:
		# 				self.max_validation_accuracy = current_accuracy
		# 				save_path = saver.save(sess, save_path)
		# 				print("Model saved in file: %s" % save_path)
				
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
		target = process_target_model_2(y, self.dict_n_grams)
		return (np.mean(predicted == target))

if __name__== '__main__':
	from .settings import training_directory
	from os.path import join as pjoin
	a = second_model()
	test = np.load(pjoin(training_directory, 'test.npy')).reshape(1,32,32,3)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		a.predict(test)
