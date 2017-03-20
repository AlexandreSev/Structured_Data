# coding: utf-8

import numpy as np
import pandas as pd 
from DSOLFUTR.settings import training_directory
from os.path import join as pjoin
from DSOLFUTR.model1 import first_model
import tensorflow as tf
from time import gmtime, strftime

data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", 
	index_col=0)

folder1 = data[data.file.apply(lambda r: (r.split('/')[1] == "1"))]

dict_conversion = {}
for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789"):
	dict_conversion[char] = i

firststep = True
for file_name, target in zip(folder1.file, folder1.tag):
	temp_name = file_name.split(".")
	temp_name = "".join(temp_name[:-1] + ["_reshaped.npy"])
	if firststep:
		list_file = np.load(pjoin(training_directory, temp_name)).reshape((1, 32, 32, 3))
		temp = list(target)
		temp = [dict_conversion[char.lower()] for char in temp 
				if char.lower() in "abcdefghijklmnopqrstuvwxyz0123456789"]
		while len(temp) != 23:
			temp.append(36)
		list_target = [temp]
		firststep = False
	else:
		list_file = np.vstack((list_file, 
					np.load(pjoin(training_directory, temp_name)).reshape((1, 32, 32, 3))))
		temp = list(target)
		temp = [dict_conversion[char.lower()] for char in temp 
				if char.lower() in "abcdefghijklmnopqrstuvwxyz0123456789"]
		while len(temp) != 23:
			temp.append(36)
		list_target.append(temp)

model = first_model(learning_rate=1e-4)

prediction = model.predict(list_file)
print(np.mean(prediction == np.array(list_target)))

one_hot_target = np.array([[np.eye(37)[i] for i in l] for l in list_target])

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(100):
		model.train_step(list_file, one_hot_target, sess)
		if i % 10 == 0:
			print(strftime("%H:%M:%S", gmtime())+" Epoch: %r"%i)
			prediction = model.predict(list_file)
			print("score: "+str(np.mean(prediction == np.array(list_target))))

	save_path = saver.save(sess, "./model.ckpt")
	print("Model saved in file: %s" % save_path)
	# https://www.tensorflow.org/programmers_guide/variables#restoring_variables

with tf.Session() as sess:
	saver.restore(sess, "./model.ckpt")
	print("Model Loaded.")

