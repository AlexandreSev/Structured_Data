#coding:utf-8

import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle

from .convolution_part import convolutional_part
from ..utils.ngrams import get_ngrams, get_dict_ngrams
from ..utils.utils import weight_variable, bias_variable

class second_model():
	"""
	"""

	def __init__(self, input_shape=(None, 32, 32, 1), learning_rate=1e-4, cnn=None):
		
		self.ngrams = get_ngrams()

		self.output_size = len(self.ngrams)

		if cnn is None:
			self.cnn = convolutional_part(input_shape)
		else:
			self.cnn = cnn

		self.W_h = weight_variable(shape=(4096, 1024))
		self.b_h = bias_variable(shape=[1024])
		
		self.input = tf.reshape(self.cnn.maxpool3, [-1, 4096])

		self.h = tf.nn.relu(tf.matmul(self.input, self.W_h) + self.b_h)

		self.dropouted_h = tf.nn.dropout(self.h, 0.7)

		self.W_o = weight_variable(shape=(1024, self.output_size))
		self.b_o = bias_variable(shape=[self.output_size])

		self.output = tf.sigmoid(tf.matmul(self.dropouted_h, self.W_o) + self.b_o) 


		self.target = tf.placeholder(tf.float32, shape=(None, self.output_size)) 

		self.create_train(learning_rate=learning_rate)

		self.max_validation_accuracy = 0
	
	def predict_proba(self, x, sess):
		feed_dict = {self.cnn.x: x}
		return sess.run(self.output, feed_dict=feed_dict)

	def predict(self, x, sess, treshold=0.5):
		feed_dict = {self.cnn.x: x}
		predicted = sess.run(self.output, feed_dict=feed_dict)
		return (predicted > treshold).astype(np.int)

	def create_train(self, learning_rate=0.001):
		self.loss = tf.nn.l2_loss(self.output - self.target)
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		

	def f_train_step(self, x, target, sess):
		feed_dict = {self.cnn.x: x, self.target: target}
		loss_score = (sess.run(self.loss, feed_dict=feed_dict))
		sess.run(self.train_step, feed_dict=feed_dict)
		print("Loss: %s"%loss_score)

	def load_weights(self, weights_path, sess):
		saver = tf.train.Saver()
		saver.restore(sess, weights_path)
		print("Model Loaded.")

	def train(self, x, target, sess, nb_epoch=100, save=True, warmstart=False, 
			  weights_path="./model2.ckpt", save_path="./model2.ckpt", test_x=None, 
			  test_target=None):
		
		print( "%s training pictures"%x.shape[0])
		print( "%s testing pictures"%test_x.shape[0])

		print("Goal on the training set: %s"%np.mean(target == 0))
		print("Goal on the testing set: %s"%np.mean(test_target == 0))

		prediction = self.predict(x, sess)
		print("Initial accuracy: %s"%np.mean(prediction == np.array(target)))

		saver = tf.train.Saver()

		if warmstart:
			saver.restore(sess, weights_path)
			print("Model Loaded.")
		
		accuracy = []
		for i in range(1, nb_epoch + 1):

			self.f_train_step(x, target, sess)
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			
			if i % 5 == 0:
				print("Training accuracy: %s" %self.compute_accuracy(x, target, sess))
				if test_x is not None:
					current_accuracy = self.compute_accuracy(test_x, test_target, sess)
					print("Validation accuracy: %s" %current_accuracy)
					if current_accuracy > self.max_validation_accuracy:
						self.max_validation_accuracy = current_accuracy
						save_path = saver.save(sess, save_path)
						print("Model saved in file: %s" % save_path)
						with open('accuracy_2.pickle', 'wb') as file:
							pickle.dump(accuracy, file, protocol=pickle.HIGHEST_PROTOCOL)
					
				accuracy.append(np.mean(prediction == target))
				

	def compute_accuracy(self, x, target, sess):
		predicted = self.predict(x, sess)
		return (np.mean(predicted == target))

if __name__== '__main__':
	from .settings import training_directory
	from os.path import join as pjoin
	a = second_model()
	test = np.load(pjoin(training_directory, 'test.npy')).reshape(1,32,32,3)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		a.predict(test)
