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

'''
Script to compute Resnet representations on oxford data.
Goal is to:
- Load the resnet network
- Cut the last fully connected layer (aimed at classifying initially)
- Compute representations
'''

training_directory = "/mnt/oxford_data/ramdisk/max/90kDICT32px"
representations_directory = "/home/antoine/representations"
targets_directory = "/home/antoine/targets/"

#training_directory_word = pjoin(training_directory, "word")
#training_directory_rpz = pjoin(training_directory, "representations")

model = ResNet50(include_top=False, weights='imagenet', input_shape=(197, 197, 3))

batches_dirs = [pjoin(training_directory, path) for path in os.listdir(training_directory) if '.txt' not in path and '.DS_Store' not in path]
batches_dirs.sort()
for batch_dir in batches_dirs: # Loop through 1..3000 folders
	batch_nb = batch_dir.split('/')[-1]
	try:
		if os.path.isfile(pjoin(representations_directory, "img_emb_" + batch_nb + '.h5')) &  os.path.isfile(pjoin(targets_directory, "target_" + batch_nb + '.txt')):
			continue

		subbatches_dirs = [pjoin(batch_dir, path) for path in os.listdir(batch_dir) if '.DS_Store' not in path]
		if not subbatches_dirs:
			print("batch", batch_nb, "empty")
			continue	
		subbatches_dirs.sort()

		imgs = []
		target = []
		for subbatch_dir in subbatches_dirs:
			subbatch_nb = subbatch_dir.split('/')[-1]

			files = [img for img in os.listdir(subbatch_dir) if 'jpg' in img and '.DS_Store' not in img]
			files.sort()
			if not files: continue

			for file_name in files: # Load images
				img = imread(pjoin(subbatch_dir, file_name))

				# Reshape images so it fits the minimum required by ResNet50#
				img = imresize(img, (197,197)).astype("float32") 
				img = preprocess_input(img[np.newaxis])
				imgs.append(img)
				target.append(file_name.split('_')[1])
		batch_tensor = np.vstack(imgs)

		# Get representations
		out_tensor = model.predict(batch_tensor, batch_size=256)
		out_tensor = out_tensor.reshape((-1, out_tensor.shape[-1]))

	    # Serialize representations
		h5f = h5py.File(pjoin(representations_directory, "img_emb_" + batch_nb + '.h5'), 'w')
		h5f.create_dataset('img_emb', data=out_tensor)
		h5f.close()

		# Serialize targets
		with open(pjoin(targets_directory, "target_" + batch_nb + '.txt'), "wb") as fp:   #Pickling
			pickle.dump(target, fp)

		print("batch", batch_nb)
	except:
		print("Error on batch %s"%batch_nb)
