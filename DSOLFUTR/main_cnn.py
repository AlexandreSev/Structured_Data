import numpy as np
from utils.settings import training_directory
from os.path import join as pjoin
from models.cnn import convolutional_part
if __name__ == "__main__":
	test = np.load(pjoin(training_directory, "word/1/1_reshaped.npy")).reshape(1, 197, 197, 3)
	model = convolutional_part()
	print(model.predict(test))