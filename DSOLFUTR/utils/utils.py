# coding: utf-8

from .settings import training_directory
from .ngrams import get_dict_ngrams, get_ngrams 

import pandas as pd
import numpy as np
from os.path import join as pjoin
import tensorflow as tf 

from sklearn.model_selection import train_test_split


def get_one_folder(n):
	"""
	Load the folder n from the ICDAR dataset
	"""
	data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", index_col=0)
	return data[data.file.apply(lambda r: (r.split('/')[1] == str(n)))]


def create_conversion_model1():
	"""
	Create the dictionnay to convert a character into its position in the output of model 1
	"""
	dict_conversion = {}
	for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789_"):
		dict_conversion[char] = i
	return dict_conversion

def process_target_model_1(target, dict_conversion):
	"""
	Change a list of labels into a list of list of number
	"""
	list_target = []

	for tag in target:

		temp = list(tag)
		temp = [dict_conversion[char.lower()] for char in temp 
				if char.lower() in "abcdefghijklmnopqrstuvwxyz0123456789"]

		while len(temp) != 23:
			temp.append(36)

		list_target.append(temp)

	return np.array(list_target)

def process_target_model_2(targets, dict_conversion):
	"""
	Compute the sparse representation of each target
	"""
	list_target = np.zeros((len(targets), len(dict_conversion.keys())))

	for nrow, target in enumerate(targets):
		target_completed = target
		while len(target_completed) != 23:
			target_completed += "_"
		list_tag = [dict_conversion.get(i, None) for i in get_ngrams(target_completed)]

		for i in list_tag:
			if i is not None:
				list_target[nrow, i] = 1

	return list_target

def process_file_name(file_names, shape=(1, 32, 32, 1)):
	"""
	From a list of file_name, load all the pictures in memory
	"""
	firststep = True
	for file_name in file_names:
		temp_name = file_name.split(".")
		temp_name = pjoin(training_directory, "".join(temp_name[:-1] + ["_reshaped.npy"]))
		if firststep:
			list_file = np.load(temp_name).reshape(shape)
			firststep = False
		else:
			list_file = np.vstack((list_file, np.load(temp_name).reshape(shape)))
	return list_file

def preprocess_data(folder, n_model=1, shape=(1, 32, 32, 1)):
	"""
	From a folder number (or None for all folders) and the number of the model, return a 
	training and a testing set
	"""
	list_file = process_file_name(folder.file, shape)

	if n_model == 1:
		dict_conversion = create_conversion_model1()
		list_target = process_target_model_1(folder.tag, dict_conversion)
	elif n_model == 2:
		list_ngrams = get_ngrams()
		dict_conversion = get_dict_ngrams(list_ngrams)
		list_target = process_target_model_2(folder.tag, dict_conversion)
	elif n_model == 3:
		list_target = np.array(folder.tag)
	else:
		raise ValueError

	train_file, test_file, train_target, test_target = train_test_split(list_file, list_target)
	
	return train_file, test_file, train_target, test_target 

def one_hot(list_target, n=37):
	"""
	return one hot encoding of a label
	"""
	return np.array([[np.eye(n)[i] for i in l] for l in list_target])

def get_score(prediction2, dico_conversion, x):
	"""
	Return the score of the string x obtained in the prediction 2 and 0.5 if x is not predicted
	"""
	if x in dico_conversion:
		return prediction2[dico_conversion[x]]
	else:
		return 0.5

def build_plot(pathnpyfile, Xaxis="Epochs", Yaxis="Loss"):
	"""
	Build the plot of Yaxis versus Xaxis
	"""
	list_to_be_plotted = np.load(pathnpyfile)
	pd_list = pd.DataFrame(list_to_be_plotted.T, columns=[Xaxis, Yaxis])
	# pd_list.plot()
	sns.set_style("darkgrid")
	plt.plot(pd_list[Xaxis], pd_list[Yaxis], label=Yaxis)
	#seabornplot = sns.tsplot(data=pd_list, time = Xaxis, value=Yaxis)
	plt.legend()

	plt.savefig(r"output_"+Yaxis+".png") #time='Date', unit='Dummy', condition='Company', value='Price'
	plt.show()


def visualize(prediction, dict_inverse):
	"""
	Transform results from a list of number to a list of strings
	"""
	output = []
	for word in range(len(prediction)):
		outputword = "".join([dico_inverse[i] for i in prediction[word]])
		output.append(outputword)
	return output
	