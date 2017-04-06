import numpy as np 
import pandas as pd
from settings import training_directory
from skimage.transform import resize
from skimage.io import imread 
from skimage.color import rgb2grey
import os

"""
Script to resize all pictures of ICDAR dataset
"""

word = pd.read_csv(os.path.join(training_directory, "word.csv"), sep=";", index_col=0)

for file in word.file:
	path_img = os.path.join(training_directory, str(file))
	img = imread(path_img)
	print(img.shape)
	img_reshaped = resize(image= img, output_shape=(197, 197, 3))
	np.save(os.path.join(training_directory,file[:-4]+"_reshaped.npy"),img_reshaped)