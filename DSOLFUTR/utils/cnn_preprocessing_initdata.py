from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread, imresize
import numpy as np
import h5py
import os
from os.path import join as pjoin

'''
Script to compute Resnet representations on oxford data.
Goal is to:
- Load the resnet network
- Cut the last fully connected layer (aimed at classifying initially)
- Compute representations
'''

# Execute the script from the models directory
training_directory = ""
training_directory_word = pjoin(training_directory, "word")
training_directory_rpz = pjoin(training_directory, "representations")

model = ResNet50(include_top=False, weights='imagenet', input_shape=(197, 197, 3))

dir_paths = [pjoin(training_directory_word, path) for path in os.listdir(training_directory_word) if '.DS_Store' not in path]
for dir_path in dir_paths: # Loop through 1..12 folders
	n_batch = dir_path.split("/")[-1]
	files = [img for img in os.listdir(dir_path) if "reshaped" not in img]
	files.sort()

	files_int = [int(img.split(".")[0]) for img in files]
	order_files = np.argsort(files_int)
	files_sorted = [files[i] for i in order_files]
	imgs = []
	for file_name in files: # Load images
		img = imread(pjoin(dir_path, file_name))

		# Reshape images so it fits the minimum required by ResNet50#
		img = imresize(img, (197,197)).astype("float32") 
		img = preprocess_input(img[np.newaxis])
		imgs.append(img)
	batch_tensor = np.vstack(imgs)
	print("batch:", n_batch, "tensor shape:", batch_tensor.shape)

	# Get representations
	out_tensor = model.predict(batch_tensor, batch_size=32)
	out_tensor = out_tensor.reshape((-1, out_tensor.shape[-1]))
	print("output shape:", out_tensor.shape)

    # Serialize representations
	h5f = h5py.File(pjoin(training_directory_rpz, "img_emb_" + n_batch + '.h5'), 'w')
	h5f.create_dataset('img_emb', data=out_tensor)
	h5f.close()