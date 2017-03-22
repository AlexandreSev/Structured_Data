# coding: utf-8

from .settings import training_directory
from .ngrams import get_dict_ngrams, get_ngrams

import pandas as pd
import numpy as np
from os.path import join as pjoin

from sklearn.model_selection import train_test_split


def get_one_folder(n):
	data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", index_col=0)
	return data[data.file.apply(lambda r: (r.split('/')[1] == str(n)))]


def create_conversion_model1():
	dict_conversion = {}
	for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789"):
		dict_conversion[char] = i
	return dict_conversion

def process_target_model_1(target, dict_conversion):
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

	list_target = np.zeros((len(targets), len(dict_conversion.keys())))

	for nrow, target in enumerate(targets):
		list_tag = [dict_conversion.get(i, None) for i in get_ngrams(target)]

		for i in list_tag:
			if i is not None:
				list_target[nrow, i] = 1

	return list_target

def process_file_name(file_names):
	firststep = True
	for file_name in file_names:
		temp_name = file_name.split(".")
		temp_name = pjoin(training_directory, "".join(temp_name[:-1] + ["_reshaped.npy"]))
		if firststep:
			list_file = np.load(temp_name).reshape((1, 32, 32, 1))
			firststep = False
		else:
			list_file = np.vstack((list_file, np.load(temp_name).reshape((1, 32, 32, 1))))
	return list_file

def preprocess_data(folder, n_model=1):

	list_file = process_file_name(folder.file)

	if n_model == 1:
		dict_conversion = create_conversion_model1()
		list_target = process_target_model_1(folder.tag, dict_conversion)
	elif n_model == 2:
		list_ngrams = get_ngrams()
		dict_conversion = get_dict_ngrams(list_ngrams)
		list_target = process_target_model_2(folder.tag, dict_conversion)
	else:
		raise ValueError

	train_file, test_file, train_target, test_target = train_test_split(list_file, list_target)
	
	return train_file, test_file, train_target, test_target 

def one_hot(list_target, n=37):
	return np.array([[np.eye(n)[i] for i in l] for l in list_target])



