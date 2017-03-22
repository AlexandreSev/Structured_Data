#coding: utf-8

import pandas as pd
import tensorflow as tf

from DSOLFUTR.utils.utils import *
from DSOLFUTR.utils.settings import training_directory

from DSOLFUTR.models import model1, model2

def main(folder=None, n_model=1, nb_epoch=100, save=True, warmstart=False, 
              weights_path="./model2.ckpt", save_path="./model2.ckpt"):

	data_path = pjoin(training_directory, "word.csv")

	if folder is None:
		data = pd.read_csv(data_path, sep=";", encoding="utf8", index_col=0)
	else:
		data = get_one_folder(folder)

	train_file, test_file, train_target, test_target = preprocess_data(data, n_model=n_model)

	with tf.Session() as sess:

		if n_model == 1:
			model = model1.first_model()
			training_target = one_hot(train_target)
		elif n_model == 2:
			model = model2.second_model()
			training_target = None
		else:
			raise ValueError

		sess.run(tf.global_variables_initializer())
		model.train(train_file, train_target, sess, nb_epoch, save, warmstart, 
					weights_path, save_path, training_target, test_file, test_target)


if __name__ == "__main__":
	main(n_model=2)

