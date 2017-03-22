# coding: utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0., shape=shape)
	return tf.Variable(initial)



class first_model():

	
	def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4):
		self.cnn = convolutional_part(input_shape)

		self.W = np.array([weight_variable((4096, 37)) for _ in range(23)])
		self.b = np.array([bias_variable([37]) for _ in range(23)])
		self.input = tf.reshape(self.cnn.maxpool3, [-1, 4096])

		self.output = np.array([tf.nn.softmax(tf.matmul(self.input, self.W[k]) + self.b[k], dim=-1) 
								for k in range(23)])
		self.target = np.array([tf.placeholder(tf.float32, shape=(None, 37)) 
								for k in range(23)])

		self.create_train(learning_rate=learning_rate)
		

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


	def train(self, x, target, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model1.ckpt", save_path="./model1.ckpt", training_target=None,
			  test_x=None, test_target=None):

		print( "%s training pictures"%x.shape[0])
		print( "%s testing pictures"%test_x.shape[0])

		prediction = self.predict(x, sess)
		print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		saver = tf.train.Saver()

		if warmstart:
			saver.restore(sess, weights_path)
			print("Model Loaded.")
		
		accuracy = []
		for i in range(1, nb_epoch + 1):

			self.f_train_step(x, training_target, sess)
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 10 == 0:
				print("Training accuracy: %s" %self.compute_accuracy(x, target, sess))
				if test_x is not None:
					print("Validation accuracy: %s" %self.compute_accuracy(test_x, test_target, sess))
				accuracy.append(np.mean(prediction == target))

			if save & (i % 50 == 0):
				save_path = saver.save(sess, save_path)
				print("Model saved in file: %s" % save_path)
				with open('accuracy_1.pickle', 'wb') as file:
					pickle.dump(accuracy, file, protocol=pickle.HIGHEST_PROTOCOL)

	def compute_accuracy(self, x, target, sess):
		predicted = self.predict(x, sess)
		return (np.mean(predicted == target))
