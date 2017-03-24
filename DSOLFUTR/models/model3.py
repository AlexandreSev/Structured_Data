# coding: utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part
from .model1 import first_model
from .model2 import second_model
from ..utils.beamsearch import beam_search
from ..utils.ngrams import get_dict_ngrams, reverse_dict
from ..utils.utils import *


class hybrid_model():

	def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4):

		self.list_character = "abcdefghijklmnopqrstuvwxyz0123456789_"
		self.dict_conversion = create_conversion_model1()

		self.cnn = convolutional_part(input_shape)
		self.first_model = first_model(input_shape=(None, 32, 32, 1), learning_rate=1e-4, 
										cnn=self.cnn)
		self.second_model = second_model(input_shape=(None, 32, 32, 1), learning_rate=1e-4, 
										cnn=self.cnn)

		self.dict_ngrams = get_dict_ngrams(self.second_model.ngrams)
		self.reverse_dict_ngrams = reverse_dict(self.dict_ngrams)

		self.max_validation_accuracy = 0

	def predict(self, x, sess):
		
		prediction1 = self.first_model.predict_proba(x, sess)
		prediction2 = self.second_model.predict_proba(x, sess)

		response = []
		for nrow in range(x.shape[0]):
			result = beam_search(prediction1[nrow, 0, :], prediction2[nrow], self.list_character, self.dict_ngrams,
								 self.dict_conversion)
			response.append(result)

		return response

	def f_train_step(self, x, target1, target2, sess):
		loss_score1 = 0
		for k in range(23):
			feed_dict = {self.cnn.x: x, self.first_model.target[k]: target1[:, k, :]}
			temp = - tf.log(tf.reduce_sum(tf.multiply(self.first_model.output[k], self.first_model.target[k]), axis=1))
			loss_score1 += (sess.run(tf.reduce_sum(temp), feed_dict=feed_dict))
			sess.run(self.first_model.train_steps[k], feed_dict=feed_dict)
		print("Loss model 1: %s"%loss_score1)
		feed_dict = {self.cnn.x: x, self.second_model.target: target2}
		loss_score2 = (sess.run(self.second_model.loss, feed_dict=feed_dict))
		sess.run(self.second_model.train_step, feed_dict=feed_dict)
		print("Loss model 2: %s"%loss_score2)
		print("Loss: %s"%(loss_score1 + loss_score2))

	def train(self, x, target, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model3.ckpt", save_path="./model3.ckpt", test_x=None, 
			  test_target=None):
		
		print( "%s training pictures"%x.shape[0])
		print( "%s testing pictures"%test_x.shape[0])

		print("Goal on the training set: %s"%np.mean(target == 0))
		print("Goal on the testing set: %s"%np.mean(test_target == 0))

		target_model_1 = process_target_model_1(target, self.dict_conversion)
		training_target_model_1 = one_hot(target_model_1)
		target_model_2 = process_target_model_2(target, self.dict_ngrams)

		test_target_m1 = process_target_model_1(test_target, self.dict_conversion)

		prediction = self.predict(x, sess)
		print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		tmp = self.first_model.compute_accuracy(x, target_model_1, sess)
		print("Initial accuracy character per character: %s"%tmp )

		saver = tf.train.Saver()

		if warmstart:
			saver.restore(sess, weights_path)
			print("Model Loaded.")
		
		accuracy = []
		for i in range(1, nb_epoch + 1):

			self.f_train_step(x, training_target_model_1, target_model_2, sess)
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 5 == 0:
				tmp = self.first_model.compute_accuracy(x, target_model_1, sess)
				print("Training accuracy character per character: %s"%tmp )
				print("Training accuracy: %s" %self.compute_accuracy(x, target, sess))
				if test_x is not None:
					tmp = self.compute_accuracy(test_x, test_target, sess)
					print("Validation accuracy: %s" %tmp)
					current_accuracy = self.first_model.compute_accuracy(test_x, test_target_m1, sess)
					print("Testing accuracy character per character: %s"%current_accuracy )
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)
						with open('accuracy_3.pickle', 'wb') as file:
							pickle.dump(accuracy, file, protocol=pickle.HIGHEST_PROTOCOL)
					
				#accuracy.append(np.mean(prediction == target))

	def compute_accuracy(self, x, target, sess):
		predicted = self.predict(x, sess)
		return (np.sum([predicted[i] == target[i] for i in range(len(predicted))]) * 1. / len(predicted))









