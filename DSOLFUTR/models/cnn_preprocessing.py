from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread, imresize
import numpy as np
import h5py
import os

# Execute the script from the models directory
training_directory = "../data/"
training_directory_word = training_directory + "word/"
training_directory_rpz = training_directory + "representations/"

model = ResNet50(include_top=False, weights='imagenet', input_shape=(197, 197, 3))

dir_paths = [training_directory_word + path for path in os.listdir(training_directory_word)]
for i, dir_path in enumerate(dir_paths): # Loop through 1..12 folders
	files = [img for img in os.listdir(dir_path) if "reshaped" not in img]
	imgs = []
	for file in files: # Load images
		img = imread(dir_path + '/' + file)

		# Reshape images so it fits the minimum required by ResNet50
		img = imresize(img, (197,197)).astype("float32") 
		img = preprocess_input(img[np.newaxis])
		imgs.append(img)
	batch_tensor = np.vstack(imgs)
	print("batch:", i, "tensor shape:", batch_tensor.shape)

	# Get representations
	out_tensor = model.predict(batch_tensor, batch_size=32)
	print("output shape:", out_tensor.shape)

    # Serialize representations
	h5f = h5py.File(training_directory + 'representations/img_emb_' + str(i+1) + '.h5', 'w')
	h5f.create_dataset('img_emb_' + str(i+1), data=out_tensor)
	h5f.close()