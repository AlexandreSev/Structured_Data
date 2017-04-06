#coding: utf-8

import pandas as pd
import tensorflow as tf

from DSOLFUTR.utils.utils import *
from DSOLFUTR.utils.settings import training_directory

from DSOLFUTR.models import model1, model2, model3, cnn, model1_resnet, model2_resnet, model1_resnet_ox, model2_resnet_ox

def main(folder=None, n_model=1, nb_epoch=100, save=True, warmstart=False, 
         weights_path="./model1.ckpt", save_path="./model1.ckpt", learning_rate=10e-3, 
		input_shape=(None, 32, 32, 1)):
	"""

	data_path = pjoin(training_directory, "word.csv")

	if folder is None:
		data = pd.read_csv(data_path, sep=";", encoding="utf8", index_col=0)
	else:
		data = get_one_folder(folder)

	train_file, test_file, train_target, test_target = preprocess_data(data, n_model=n_model, shape=input_shape)

	with tf.Session() as sess:
		if n_model == 1:
			model = model1.first_model(cnn=cnn.resnet(), input_shape=input_shape)
		elif n_model == 2:
			model = model2.second_model(input_shape=input_shape)
		elif n_model == 3:
			model = model3.hybrid_model()
		else:
			raise ValueError
	"""
	with tf.Session() as sess:
		#model = model1_resnet_ox.first_head(learning_rate=learning_rate)
		model = model2_resnet_ox.second_head(learning_rate=learning_rate)

		sess.run(tf.global_variables_initializer())
		
		list_dir = range(1000, 1601)
		test_dir = range(1601, 1701)
		model.train(list_dir, sess, nb_epoch, save, warmstart, 
					weights_path, save_path, test_dir)

import sys
if __name__ == "__main__":
	if len(sys.argv) > 1: input_shape=tuple(None, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) 
	else: input_shape= (None, 32, 32, 1)
	main(n_model=1, folder=1, weights_path="./model3.ckpt", save_path="./model3.ckpt", warmstart=False,
		input_shape=input_shape)

