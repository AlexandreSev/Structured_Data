from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread, imresize
import numpy as np
import h5py
import os
import pickle
from os.path import join as pjoin

training_directory = "/mnt/oxford_data/ramdisk/max/90kDICT32px"
representations_directory = "/mnt/oxford_data/representations"
targets_directory = "/mnt/oxford_data/targets/"

training_directory = "/Users/antoine/Downloads/mnt/ramdisk/max/90kDICT32px"
representations_directory = "/Users/antoine/Downloads/mnt/ramdisk/max/representations"
targets_directory = "/Users/antoine/Downloads/mnt/ramdisk/max/targets"

#training_directory_word = pjoin(training_directory, "word")
#training_directory_rpz = pjoin(training_directory, "representations")

model = ResNet50(include_top=False, weights='imagenet', input_shape=(197, 197, 3))

batches = [pjoin(training_directory, path) for path in os.listdir(training_directory) if '.txt' not in path and '.DS_Store' not in path]
for batch in batches: # Loop through 1..3000 folders
	
	subbatches = [pjoin(batch, path) for path in os.listdir(batch) if '.DS_Store' not in path]
	for subbatch in subbatches:

		files = [img for img in os.listdir(subbatch) if 'jpg' in img and '.DS_Store' not in img]
		files.sort()

		#files_int = [int(img.split(".")[0]) for img in files]
		#order_files = np.argsort(files_int)
		#files_sorted = [files[i] for i in order_files]
		imgs = []
		target = []
		for file_name in files: # Load images
			img = imread(pjoin(subbatch, file_name))

			# Reshape images so it fits the minimum required by ResNet50#
			img = imresize(img, (197,197)).astype("float32") 
			img = preprocess_input(img[np.newaxis])
			imgs.append(img)
			target.append(file_name.split('_')[1])
		batch_tensor = np.vstack(imgs)
		print('coucou')

		# Get representations
		out_tensor = model.predict(batch_tensor, batch_size=256)
		out_tensor = out_tensor.reshape((-1, out_tensor.shape[-1]))
		print("batch", batch, "subbatch", subbatch)

	    # Serialize representations
		h5f = h5py.File(pjoin(representations_directory, "img_emb_" + batch + '_' + subbatch + '.h5'), 'w')
		h5f.create_dataset('img_emb', data=out_tensor)
		h5f.close()

		with open(pjoin(targets_directory, "target_" + batch + '_' + subbatch + '.txt'), "wb") as fp:   #Pickling
			pickle.dump(target, fp)