# coding: utf-8

import numpy as np
import pandas as pd 
from DSOLFUTR.settings import training_directory
from os.path import join as pjoin
from DSOLFUTR.model1 import first_model

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
		print(temp)
		list_target.append(temp)

model = first_model()

prediction = model.predict(list_file)
print(np.mean(prediction == np.array(list_target)))


