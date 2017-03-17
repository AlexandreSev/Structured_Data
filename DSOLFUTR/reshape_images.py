import numpy as np 
import pandas as pd
from settings import training_directory
from skimage.transform import resize
from skimage.io import imread 
import os

word = pd.read_csv(os.path.join(training_directory, "word.csv"), sep=";", index_col=0)


for file in word.file:
	path_img = os.path.join(training_directory, str(file))
	img = imread(path_img)
	img_reshaped = resize(image= img, output_shape=(32, 32, 3))
	np.save(os.path.join(training_directory,file[:-4]+"_reshaped.npy"),img_reshaped)
