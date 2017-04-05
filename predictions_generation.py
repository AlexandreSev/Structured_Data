import sys
import h5py
if sys.argv[1] == '1': from DSOLFUTR.models.model1_resnet_ox import first_head
elif sys.argv[1] == '2': from DSOLFUTR.models.model2_resnet import second_head
import tensorflow as tf

model1_weights_path='./model1.ckpt'
model2_weights_path='./model4.ckpt'

h5f = h5py.File('img_emb_1.h5', 'r')
representations = h5f['img_emb'][:]
h5f.close()
print(representations)

with tf.Session() as sess:
	if sys.argv[1] == '1':
		model = first_head()
		model.load_weights(model1_weights_path, sess)
		predictions = model.predict(representations, sess)
		h5f = h5py.File('preds_model1.h5', 'w')
		h5f.create_dataset('model1', data=predictions)
	elif sys.argv[1] == '2':
		model = second_head()
		model.load_weights(model2_weights_path, sess)
		predictions = model.predict(representations, sess)
		h5f = h5py.File('preds_model2.h5', 'w')
		h5f.create_dataset('model2', data=predictions)
	h5f.close()