#coding: utf- 8

from DSOLFUTR.utils.settings import training_directory
from os.path import join as pjoin
from DSOLFUTR.models.model2 import second_model
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from DSOLFUTR.utils.ngrams import get_dict_ngrams, get_ngrams
import pandas as pd
from time import gmtime, strftime
import pickle

data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", 
	index_col=0)

folder1 = data[data.file.apply(lambda r: (r.split('/')[1] == "1"))]

list_ngrams = get_ngrams()

dict_conversion = get_dict_ngrams(list_ngrams)


firststep = True

list_target = np.zeros((len(folder1), len(list_ngrams)))

for nrow, file_name, target in zip(range(len(folder1.file)), folder1.file, folder1.tag):
	temp_name = file_name.split(".")
	temp_name = "".join(temp_name[:-1] + ["_reshaped.npy"])
	if firststep:
		list_file = np.load(pjoin(training_directory, temp_name)).reshape((1, 32, 32, 1))
		firststep = False
	else:
		list_file = np.vstack((list_file, 
					np.load(pjoin(training_directory, temp_name)).reshape((1, 32, 32, 1))))
	list_tag = [ dict_conversion.get(i, None) for i in get_ngrams(target)]
	for i in list_tag:
		if i is not None:
			list_target[nrow, i] = 1


with tf.Session() as sess:
	model = second_model(learning_rate=1e-4)
	sess.run(tf.global_variables_initializer())

	prediction = model.predict(list_file, sess)
	print(np.mean(prediction == np.array(list_target)))

	saver = tf.train.Saver()

	if True:
		saver.restore(sess, "./model2.ckpt")
		print("Model Loaded.")
	else:
		sess.run(tf.global_variables_initializer())
	
	accuracy = []
	for i in range(1, 100000):
		model.f_train_step(list_file, list_target, sess)
		print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
		if i % 10 == 0:
			prediction = model.predict(list_file, sess)
			print("accuracy: "+str(np.mean(prediction == list_target)))
			accuracy.append(np.mean(prediction == list_target))
		if i % 50 == 0:
			save_path = saver.save(sess, "./model2.ckpt")
			print("Model saved in file: %s" % save_path)
			with open('accuracy_2.pickle', 'wb') as file:
				pickle.dump(accuracy, file, protocol=pickle.HIGHEST_PROTOCOL)
				